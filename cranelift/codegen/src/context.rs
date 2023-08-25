//! Cranelift compilation context and main entry point.
//!
//! When compiling many small functions, it is important to avoid repeatedly allocating and
//! deallocating the data structures needed for compilation. The `Context` struct is used to hold
//! on to memory allocations between function compilations.
//!
//! The context does not hold a `TargetIsa` instance which has to be provided as an argument
//! instead. This is because an ISA instance is immutable and can be used by multiple compilation
//! contexts concurrently. Typically, you would have one context per compilation thread and only a
//! single ISA instance.

use std::collections::{HashMap, HashSet};

use crate::alias_analysis::AliasAnalysis;
use crate::dce::do_dce;
use crate::dominator_tree::DominatorTree;
use crate::egraph::EgraphPass;
use crate::flowgraph::ControlFlowGraph;
use crate::ir::{Block, BlockCall, Function, InstructionData, JumpTableData, Opcode, Value};
use crate::isa::TargetIsa;
use crate::legalizer::simple_legalize;
use crate::loop_analysis::LoopAnalysis;
use crate::machinst::{CompiledCode, CompiledCodeStencil};
use crate::nan_canonicalization::do_nan_canonicalization;
use crate::remove_constant_phis::do_remove_constant_phis;
use crate::result::{CodegenResult, CompileResult};
use crate::settings::{FlagsOrIsa, OptLevel};
use crate::trace;
use crate::unreachable_code::eliminate_unreachable_code;
use crate::verifier::{verify_context, VerifierErrors, VerifierResult};
use crate::{timing, CompileError};
#[cfg(feature = "souper-harvest")]
use alloc::string::String;
use alloc::vec::Vec;
use cranelift_control::ControlPlane;
use cranelift_entity::ListPool;

#[cfg(feature = "souper-harvest")]
use crate::souper_harvest::do_souper_harvest;

/// Persistent data structures and compilation pipeline.
pub struct Context {
    /// The function we're compiling.
    pub func: Function,

    /// The control flow graph of `func`.
    pub cfg: ControlFlowGraph,

    /// Dominator tree for `func`.
    pub domtree: DominatorTree,

    /// Loop analysis of `func`.
    pub loop_analysis: LoopAnalysis,

    /// Result of MachBackend compilation, if computed.
    pub(crate) compiled_code: Option<CompiledCode>,

    /// Flag: do we want a disassembly with the CompiledCode?
    pub want_disasm: bool,
}

impl Context {
    /// Allocate a new compilation context.
    ///
    /// The returned instance should be reused for compiling multiple functions in order to avoid
    /// needless allocator thrashing.
    pub fn new() -> Self {
        Self::for_function(Function::new())
    }

    /// Allocate a new compilation context with an existing Function.
    ///
    /// The returned instance should be reused for compiling multiple functions in order to avoid
    /// needless allocator thrashing.
    pub fn for_function(func: Function) -> Self {
        Self {
            func,
            cfg: ControlFlowGraph::new(),
            domtree: DominatorTree::new(),
            loop_analysis: LoopAnalysis::new(),
            compiled_code: None,
            want_disasm: false,
        }
    }

    /// Clear all data structures in this context.
    pub fn clear(&mut self) {
        self.func.clear();
        self.cfg.clear();
        self.domtree.clear();
        self.loop_analysis.clear();
        self.compiled_code = None;
        self.want_disasm = false;
    }

    /// Returns the compilation result for this function, available after any `compile` function
    /// has been called.
    pub fn compiled_code(&self) -> Option<&CompiledCode> {
        self.compiled_code.as_ref()
    }

    /// Set the flag to request a disassembly when compiling with a
    /// `MachBackend` backend.
    pub fn set_disasm(&mut self, val: bool) {
        self.want_disasm = val;
    }

    /// Compile the function, and emit machine code into a `Vec<u8>`.
    ///
    /// Run the function through all the passes necessary to generate
    /// code for the target ISA represented by `isa`, as well as the
    /// final step of emitting machine code into a `Vec<u8>`. The
    /// machine code is not relocated. Instead, any relocations can be
    /// obtained from `compiled_code()`.
    ///
    /// Performs any optimizations that are enabled, unless
    /// `optimize()` was already invoked.
    ///
    /// This function calls `compile`, taking care to resize `mem` as
    /// needed.
    ///
    /// Returns information about the function's code and read-only
    /// data.
    pub fn compile_and_emit(
        &mut self,
        isa: &dyn TargetIsa,
        mem: &mut Vec<u8>,
        ctrl_plane: &mut ControlPlane,
    ) -> CompileResult<&CompiledCode> {
        let compiled_code = self.compile(isa, ctrl_plane)?;
        mem.extend_from_slice(compiled_code.code_buffer());
        Ok(compiled_code)
    }

    /// Internally compiles the function into a stencil.
    ///
    /// Public only for testing and fuzzing purposes.
    pub fn compile_stencil(
        &mut self,
        isa: &dyn TargetIsa,
        ctrl_plane: &mut ControlPlane,
    ) -> CodegenResult<CompiledCodeStencil> {
        let _tt = timing::compile();

        self.verify_if(isa)?;

        self.optimize(isa)?;

        //println!("BEFORE\n{:?}", self.func);
        //self.br_tables_opt()?;
        //println!("AFTER\n{:?}", self.func);

        //self.optimize(isa)?;

        isa.compile_function(&self.func, &self.domtree, self.want_disasm, ctrl_plane)
    }

    fn br_tables_opt(&mut self) -> CodegenResult<()> {
        // Replace all unconditional jumps (jump blockn) with the block itself
        //self.compute_domtree();
        for block in self.domtree.cfg_postorder().iter().rev() {
            let inst = self.func.layout.last_inst(*block).unwrap();
            let mut old_to_new_arg = std::collections::HashMap::new();
            if let crate::ir::InstructionData::Jump {
                opcode: _,
                destination,
            } = self.func.dfg.insts[inst]
            {
                let pool = &self.func.stencil.dfg.value_lists;
                let dest_block = destination.block(pool);

                let dest_args = destination.args_slice(pool);
                let dest_params = self.func.stencil.dfg.block_params(dest_block);
                for (&k, &v) in dest_params.iter().zip(dest_args.iter()) {
                    old_to_new_arg.insert(k, v);
                }

                self.func.layout.remove_inst(inst);
                let mut inst = self.func.stencil.layout.first_inst(dest_block);
                while let Some(dest_inst) = inst {
                    let new_inst = self.func.stencil.dfg.clone_inst(dest_inst);
                    self.func
                        .stencil
                        .dfg
                        .map_inst_values(new_inst, |_, old_arg| {
                            old_to_new_arg.get(&old_arg).copied().unwrap_or(old_arg)
                        });

                    let old_res = self.func.stencil.dfg.inst_results(dest_inst);
                    let new_res = self.func.stencil.dfg.inst_results(new_inst);
                    for (&k, &v) in old_res.iter().zip(new_res.iter()) {
                        old_to_new_arg.insert(k, v);
                    }

                    self.func.stencil.layout.append_inst(new_inst, *block);
                    inst = self.func.stencil.layout.next_inst(dest_inst);
                }
            }
        }
        //self.domtree.clear();
        //self.loop_analysis.clear();
        Ok(())
    }

    /// Optimize the function, performing all compilation steps up to
    /// but not including machine-code lowering and register
    /// allocation.
    ///
    /// Public only for testing purposes.
    pub fn optimize(&mut self, isa: &dyn TargetIsa) -> CodegenResult<()> {
        log::debug!(
            "Number of CLIF instructions to optimize: {}",
            self.func.dfg.num_insts()
        );
        log::debug!(
            "Number of CLIF blocks to optimize: {}",
            self.func.dfg.num_blocks()
        );

        let opt_level = isa.flags().opt_level();
        crate::trace!(
            "Optimizing (opt level {:?}):\n{}",
            opt_level,
            self.func.display()
        );

        self.compute_cfg();
        if isa.flags().enable_nan_canonicalization() {
            self.canonicalize_nans(isa)?;
        }

        self.legalize(isa)?;

        self.compute_domtree();
        self.eliminate_unreachable_code(isa)?;

        if opt_level != OptLevel::None {
            self.dce(isa)?;
        }

        self.resolve_all_aliases();

        self.direct_dispatch_interpreter_transform()?;
        self.compute_cfg();
        self.compute_domtree();

        self.remove_constant_phis(isa)?;

        if opt_level != OptLevel::None {
            self.egraph_pass()?;
        }

        Ok(())
    }

    fn resolve_all_aliases(&mut self) {
        for block in self.domtree.cfg_postorder().iter() {
            let insts: Vec<crate::ir::Inst> = self.func.layout.block_insts(*block).collect();
            for inst in insts {
                self.func.dfg.resolve_aliases_in_arguments(inst);
            }
        }
    }

    fn get_block_params_to_add(&mut self) -> HashMap<Block, HashSet<Value>> {
        self.compute_cfg();
        self.compute_domtree();

        let mut updated_block_params_map: HashMap<Block, HashSet<Value>> = HashMap::new();
        let mut block_variable_scope: HashMap<Block, HashSet<Value>> = HashMap::new();

        // Initialize both so that we can call unwrap on get.
        for block in self.func.layout.blocks() {
            updated_block_params_map.insert(block, HashSet::new());
            block_variable_scope.insert(block, HashSet::new());
        }

        // Single pass to populate updated_block_params with variables that are used in blocks that aren't in scope.
        for block in self.domtree.cfg_postorder().iter() {
            for arg in self.func.dfg.block_params(*block) {
                block_variable_scope.get_mut(block).unwrap().insert(*arg);
            }

            for inst in self.func.stencil.layout.block_insts(*block) {
                // Check that all arguments to an instruction have arguments that are in scope.
                for arg in self.func.dfg.inst_args(inst) {
                    if !block_variable_scope.get(block).unwrap().contains(arg) {
                        updated_block_params_map
                            .get_mut(block)
                            .unwrap()
                            .insert(*arg);
                    }
                }

                // Add results of instructions to be in scope
                for result_var in self.func.dfg.inst_results(inst) {
                    block_variable_scope
                        .get_mut(block)
                        .unwrap()
                        .insert(*result_var);
                }
            }

            // The last instruction of a block might be a branch.
            // If the instruction is a branch, check that the values passed to the block are in scope.
            let last_inst = self.func.layout.last_inst(*block).unwrap();
            let branches =
                self.func.dfg.insts[last_inst].branch_destination(&self.func.dfg.jump_tables);
            for branch in branches {
                for arg in branch.args_slice(&self.func.dfg.value_lists) {
                    if !block_variable_scope.get(block).unwrap().contains(arg) {
                        updated_block_params_map
                            .get_mut(block)
                            .unwrap()
                            .insert(*arg);
                    }
                }
            }
        }

        // TODO: This is in a loop because I think I need to take a fixed point.
        // But you might be able to remove fixed point if you visit blocks in the right order?
        loop {
            let mut map_updated = false;
            for block in self.domtree.cfg_postorder().iter() {
                // Propogate new block parameters to all predecessors,
                // as long as the predecassor does not have the variables defined in their own scope.
                for pred in self.cfg.pred_iter(*block) {
                    let pred = &pred.block;
                    let new_params = updated_block_params_map.get(block).unwrap().clone();
                    for p in new_params {
                        if !block_variable_scope.get(pred).unwrap().contains(&p)
                            && !updated_block_params_map.get(pred).unwrap().contains(&p)
                        {
                            map_updated = true;
                            updated_block_params_map.get_mut(pred).unwrap().insert(p);
                        }
                    }
                }
            }
            if map_updated == false {
                break;
            }
        }

        return updated_block_params_map;
    }

    fn maximal_ssa(&mut self) {
        let new_block_param_map = self.get_block_params_to_add();

        for block in self.domtree.cfg_postorder().iter() {
            let mut old_value_to_new: HashMap<Value, Value> = HashMap::new();
            let mut new_params: Vec<Value> = new_block_param_map
                .get(&block)
                .unwrap()
                .iter()
                .cloned()
                .collect();
            new_params.sort();

            // Add each new parameter to the block paramters.
            for new_param in new_params {
                let ty = self.func.dfg.value_type(new_param);
                let new_val = self.func.dfg.append_block_param(*block, ty);
                old_value_to_new.insert(new_param, new_val);
            }

            let insts: Vec<_> = self.func.layout.block_insts(*block).collect();

            // Update all inst arguments to refer to new block parameters.
            for inst in insts {
                for inst_arg in self.func.dfg.inst_args_mut(inst) {
                    if let Some(new_val) = old_value_to_new.get(inst_arg) {
                        *inst_arg = *new_val;
                    }
                }
            }

            // If the last instruction is a branch, update block parameters to have new values
            // Also check that if the branch target block has new values that need to be passed in because of new block parameters.
            let last_inst = self.func.layout.last_inst(*block).unwrap();
            let mut inst_data = self.func.dfg.insts[last_inst].clone();
            let mut dfg = &mut self.func.dfg;
            for branch in inst_data.branch_destination_mut(&mut dfg.jump_tables) {
                for arg in branch.args_slice_mut(&mut dfg.value_lists) {
                    if let Some(new_val) = old_value_to_new.get(arg) {
                        *arg = *new_val;
                    }
                }

                let mut new_params: Vec<Value> = new_block_param_map
                    .get(&branch.block(&dfg.value_lists))
                    .unwrap()
                    .iter()
                    .cloned()
                    .collect();
                new_params.sort();

                for param in new_params {
                    let new_arg = old_value_to_new.get(&param).cloned().unwrap_or(param);
                    branch.append_argument(new_arg, &mut dfg.value_lists);
                }
            }
            self.func.dfg.insts[last_inst] = inst_data;
        }
    }

    fn check_for_loop_switch_pattern(&mut self) -> Option<Block> {
        const BR_TABLE_CASES: usize = 10;

        self.compute_loop_analysis();

        for block in self.func.layout.blocks() {
            if let Some(inst) = self.func.layout.last_inst(block) {
                if self.func.dfg.insts[inst].opcode() == Opcode::BrTable {
                    //if BR_TABLE_CASES < self.cfg.succ_iter(block).count() {
                    if self.loop_analysis.innermost_loop(block).is_some() {
                        return Some(block);
                    }
                    //}
                }
            }
        }

        None
    }

    fn direct_dispatch_interpreter_transform(&mut self) -> CodegenResult<()> {
        // 1. Find loop-switch pattern, ie, the pattern we want to transform.
        //   1.1. Do loop analysis.
        //   1.2. Check if any blocks in the loop end with a branch table instruction.
        //        This is where you can have a hauristic that filters on number of cases in the switch.
        // 2. Now that we have found pattern, apply maximal-ssa transform to the that loop.
        // 3. Direct-dispatch transform

        let mut block_map: HashMap<(Block, usize), Block> = HashMap::new();
        let mut work_q: Vec<(Block, usize)> = Vec::new();

        if let Some(br_table_block) = self.check_for_loop_switch_pattern() {
            println!("BEFORE MAX {:?}", self.func);
            self.maximal_ssa();
            println!("AFTER MAX {:?}", self.func);

            let entry_block = self.func.layout.entry_block().unwrap();
            let new_entry_block = self.func.dfg.make_block();
            self.func.layout.append_block(new_entry_block);

            work_q.push((entry_block, 0));
            block_map.insert((entry_block, 0), new_entry_block);

            let mut i = 0;
            while let Some((org_block, case_num)) = work_q.pop() {
                println!("LOOPNUM:{i} {:?}", self.func);
                i += 1;

                let mut value_map: HashMap<Value, Value> = HashMap::new();

                let new_block = block_map.get(&(org_block, case_num)).unwrap();
                println!("ORG_BLOCK:{org_block:?} NEW_BLOCK:{new_block:?} CASE:{case_num} VALUEMAP {value_map:?}");
                for old_param in self
                    .func
                    .dfg
                    .block_params(org_block)
                    .iter()
                    .cloned()
                    .collect::<Vec<_>>()
                {
                    let ty = self.func.dfg.value_type(old_param);
                    let new_param = self.func.dfg.append_block_param(*new_block, ty);
                    value_map.insert(old_param, new_param);
                    println!("OLD_P:{old_param}, NEW_P:{new_param}");
                }

                for inst in self.func.layout.block_insts(org_block).collect::<Vec<_>>() {
                    let new_inst = self.func.dfg.clone_inst(inst);
                    let mut inst_data = self.func.dfg.insts[new_inst].clone();

                    let clone_block_call = |bc: BlockCall, pool: &mut ListPool<Value>| {
                        let args = bc.args_slice(pool).iter().cloned().collect::<Vec<_>>();
                        let target = bc.block(pool);
                        BlockCall::new(target, &args[..], pool)
                    };
                    match &mut inst_data {
                        InstructionData::BranchTable { opcode, arg, table } => {
                            let default_block =
                                self.func.dfg.jump_tables[*table].default_block().clone();
                            let default_block =
                                clone_block_call(default_block, &mut self.func.dfg.value_lists);

                            let all_branches = self.func.dfg.jump_tables[*table]
                                .all_branches()
                                .iter()
                                .cloned()
                                .collect::<Vec<_>>();
                            let all_branches = all_branches
                                .into_iter()
                                .map(|b| clone_block_call(b, &mut self.func.dfg.value_lists))
                                .collect::<Vec<_>>();

                            let new_jump_table = self.func.create_jump_table(JumpTableData::new(
                                default_block,
                                all_branches.as_slice(),
                            ));

                            *table = new_jump_table;
                        }
                        _ => (),
                    }

                    self.func.dfg.insts[new_inst] = inst_data;

                    self.func.dfg.map_inst_values(new_inst, |_dfg, arg| {
                        println!("INST_ARG {arg:?}");
                        *value_map.get(&arg).unwrap()
                    });

                    for (&old_result, &new_result) in self
                        .func
                        .dfg
                        .inst_results(inst)
                        .iter()
                        .zip(self.func.dfg.inst_results(new_inst).iter())
                    {
                        value_map.insert(old_result, new_result);
                        println!("OLD_R:{old_result}, NEW_R:{new_result}");
                    }

                    self.func.layout.append_inst(new_inst, *new_block);
                }

                if let Some(last_inst) = self.func.layout.last_inst(*new_block) {
                    let mut inst_data = self.func.dfg.insts[last_inst].clone();
                    let mut dfg = &mut self.func.dfg;
                    let mut new_block_to_be_added = Vec::new();

                    for case in 0..inst_data.branch_destination(&dfg.jump_tables).len() {
                        let org_block_target = inst_data.branch_destination(&dfg.jump_tables)[case]
                            .block(&dfg.value_lists);
                        let target_case = if org_block == br_table_block {
                            case
                        } else {
                            case_num
                        };

                        let target_key = (org_block_target, target_case);

                        let new_target_block = match block_map.entry(target_key) {
                            std::collections::hash_map::Entry::Occupied(o) => *o.get(),
                            std::collections::hash_map::Entry::Vacant(v) => {
                                let new_block = dfg.make_block();
                                new_block_to_be_added.push(new_block);

                                work_q.push(target_key);
                                v.insert(new_block);
                                new_block
                            }
                        };

                        inst_data.branch_destination_mut(&mut dfg.jump_tables)[case]
                            .set_block(new_target_block, &mut dfg.value_lists);
                    }

                    for new_block_add in new_block_to_be_added {
                        self.func.layout.append_block(new_block_add);
                    }

                    self.func.dfg.insts[last_inst] = inst_data;
                }
            }

            // Delete till new_entry block since we have duplicated everything!
            while let Some(first_block) = self.func.layout.entry_block() {
                if first_block == new_entry_block {
                    break;
                }
                self.func.layout.remove_block(first_block);
            }
            println!("AFTER {:?}", self.func);
        }

        Ok(())
    }

    /// Compile the function.
    ///
    /// Run the function through all the passes necessary to generate code for the target ISA
    /// represented by `isa`. This does not include the final step of emitting machine code into a
    /// code sink.
    ///
    /// Returns information about the function's code and read-only data.
    pub fn compile(
        &mut self,
        isa: &dyn TargetIsa,
        ctrl_plane: &mut ControlPlane,
    ) -> CompileResult<&CompiledCode> {
        let stencil = self
            .compile_stencil(isa, ctrl_plane)
            .map_err(|error| CompileError {
                inner: error,
                func: &self.func,
            })?;
        Ok(self
            .compiled_code
            .insert(stencil.apply_params(&self.func.params)))
    }

    /// If available, return information about the code layout in the
    /// final machine code: the offsets (in bytes) of each basic-block
    /// start, and all basic-block edges.
    #[deprecated = "use CompiledCode::get_code_bb_layout"]
    pub fn get_code_bb_layout(&self) -> Option<(Vec<usize>, Vec<(usize, usize)>)> {
        self.compiled_code().map(CompiledCode::get_code_bb_layout)
    }

    /// Creates unwind information for the function.
    ///
    /// Returns `None` if the function has no unwind information.
    #[cfg(feature = "unwind")]
    #[deprecated = "use CompiledCode::create_unwind_info"]
    pub fn create_unwind_info(
        &self,
        isa: &dyn TargetIsa,
    ) -> CodegenResult<Option<crate::isa::unwind::UnwindInfo>> {
        self.compiled_code().unwrap().create_unwind_info(isa)
    }

    /// Run the verifier on the function.
    ///
    /// Also check that the dominator tree and control flow graph are consistent with the function.
    pub fn verify<'a, FOI: Into<FlagsOrIsa<'a>>>(&self, fisa: FOI) -> VerifierResult<()> {
        let mut errors = VerifierErrors::default();
        let _ = verify_context(&self.func, &self.cfg, &self.domtree, fisa, &mut errors);

        if errors.is_empty() {
            Ok(())
        } else {
            Err(errors)
        }
    }

    /// Run the verifier only if the `enable_verifier` setting is true.
    pub fn verify_if<'a, FOI: Into<FlagsOrIsa<'a>>>(&self, fisa: FOI) -> CodegenResult<()> {
        let fisa = fisa.into();
        if fisa.flags.enable_verifier() {
            self.verify(fisa)?;
        }
        Ok(())
    }

    /// Perform dead-code elimination on the function.
    pub fn dce<'a, FOI: Into<FlagsOrIsa<'a>>>(&mut self, fisa: FOI) -> CodegenResult<()> {
        do_dce(&mut self.func, &mut self.domtree);
        self.verify_if(fisa)?;
        Ok(())
    }

    /// Perform constant-phi removal on the function.
    pub fn remove_constant_phis<'a, FOI: Into<FlagsOrIsa<'a>>>(
        &mut self,
        fisa: FOI,
    ) -> CodegenResult<()> {
        do_remove_constant_phis(&mut self.func, &mut self.domtree);
        self.verify_if(fisa)?;
        Ok(())
    }

    /// Perform NaN canonicalizing rewrites on the function.
    pub fn canonicalize_nans(&mut self, isa: &dyn TargetIsa) -> CodegenResult<()> {
        do_nan_canonicalization(&mut self.func);
        self.verify_if(isa)
    }

    /// Run the legalizer for `isa` on the function.
    pub fn legalize(&mut self, isa: &dyn TargetIsa) -> CodegenResult<()> {
        // Legalization invalidates the domtree and loop_analysis by mutating the CFG.
        // TODO: Avoid doing this when legalization doesn't actually mutate the CFG.
        self.domtree.clear();
        self.loop_analysis.clear();

        // Run some specific legalizations only.
        simple_legalize(&mut self.func, &mut self.cfg, isa);
        self.verify_if(isa)
    }

    /// Compute the control flow graph.
    pub fn compute_cfg(&mut self) {
        self.cfg.compute(&self.func)
    }

    /// Compute dominator tree.
    pub fn compute_domtree(&mut self) {
        self.domtree.compute(&self.func, &self.cfg)
    }

    /// Compute the loop analysis.
    pub fn compute_loop_analysis(&mut self) {
        self.loop_analysis
            .compute(&self.func, &self.cfg, &self.domtree)
    }

    /// Compute the control flow graph and dominator tree.
    pub fn flowgraph(&mut self) {
        self.compute_cfg();
        self.compute_domtree()
    }

    /// Perform unreachable code elimination.
    pub fn eliminate_unreachable_code<'a, FOI>(&mut self, fisa: FOI) -> CodegenResult<()>
    where
        FOI: Into<FlagsOrIsa<'a>>,
    {
        eliminate_unreachable_code(&mut self.func, &mut self.cfg, &self.domtree);
        self.verify_if(fisa)
    }

    /// Replace all redundant loads with the known values in
    /// memory. These are loads whose values were already loaded by
    /// other loads earlier, as well as loads whose values were stored
    /// by a store instruction to the same instruction (so-called
    /// "store-to-load forwarding").
    pub fn replace_redundant_loads(&mut self) -> CodegenResult<()> {
        let mut analysis = AliasAnalysis::new(&self.func, &self.domtree);
        analysis.compute_and_update_aliases(&mut self.func);
        Ok(())
    }

    /// Harvest candidate left-hand sides for superoptimization with Souper.
    #[cfg(feature = "souper-harvest")]
    pub fn souper_harvest(
        &mut self,
        out: &mut std::sync::mpsc::Sender<String>,
    ) -> CodegenResult<()> {
        do_souper_harvest(&self.func, out);
        Ok(())
    }

    /// Run optimizations via the egraph infrastructure.
    pub fn egraph_pass(&mut self) -> CodegenResult<()> {
        let _tt = timing::egraph();

        trace!(
            "About to optimize with egraph phase:\n{}",
            self.func.display()
        );
        self.compute_loop_analysis();
        let mut alias_analysis = AliasAnalysis::new(&self.func, &self.domtree);
        let mut pass = EgraphPass::new(
            &mut self.func,
            &self.domtree,
            &self.loop_analysis,
            &mut alias_analysis,
        );
        pass.run();
        log::debug!("egraph stats: {:?}", pass.stats);
        trace!("After egraph optimization:\n{}", self.func.display());
        Ok(())
    }
}

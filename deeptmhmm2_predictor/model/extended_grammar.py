"""
Utilities for the extended-grammar CRF (first-TM vs second+-TM entry penalty).

The extended grammar adds a separate first-TM helix track (Mab1/Mba1) and
post-TM outside states (a_alpha_post/b_alpha_post), allowing the entry penalty
for the first TM crossing to be tuned independently of all subsequent crossings.

Workflow for fine-tuning
------------------------
1. Load the five trained cross-validation models as usual.
2. Call ``get_finetuning_model`` on each to obtain extended-grammar versions
   with weights transferred and the recommended second-entry penalty applied.
3. Pass the returned models to the training loop.  The models are standard
   ``SP5DeepTMHMM`` instances (with ``extend_first_tm=True``) and are fully
   compatible with the existing Lightning training infrastructure.

Example
-------
>>> from deeptmhmm2_predictor.model.extended_grammar import get_finetuning_model
>>> ext_model = get_finetuning_model(trained_model)
>>> # ext_model is ready for fine-tuning
"""

import copy
import math

from .crf import CRF
from .crf_config_with_tp_split_io import SplitIOCRFwithTPConfig


def make_extended_model(original_model, original_cfg):
    """Transfer weights from a trained model into an extended-grammar model.

    The returned model has an identical CRF grammar to the original except that
    the first TM crossing uses a separate track (Mab1/Mba1) and exits into
    post-TM outside states (a_alpha_post/b_alpha_post) rather than back to the
    plain a_alpha/b_alpha states.

    With the copied weights the Viterbi predictions are identical to the
    original model — the extension is a no-op until the entry weights are
    explicitly modified (e.g. via ``apply_extended_weights``).

    Parameters
    ----------
    original_model : SP5DeepTMHMM
        A trained model with the original (non-extended) grammar.
    original_cfg : SplitIOCRFwithTPConfig
        The CRF config of ``original_model`` (``original_model.crf_config``).

    Returns
    -------
    extended_model : SP5DeepTMHMM
        Deep-copy of ``original_model`` with the CRF replaced by the extended
        version and all weights transferred.
    extended_cfg : SplitIOCRFwithTPConfig
        The extended CRF config attached to the returned model.
    """
    ext_cfg = SplitIOCRFwithTPConfig(
        maximum_segment_lengths=original_cfg.max_segment_lengths,
        minimum_segment_lengths=original_cfg.min_segment_lengths,
        use_start_end_transitions=original_cfg.use_start_end_transitions,
        extend_first_tm=True,
    )

    device = next(original_model.parameters()).device
    N_orig = len(original_cfg.crf_states)

    m = copy.deepcopy(original_model)

    m.crf = CRF(
        len(ext_cfg.crf_states),
        batch_first=True,
        init_weight=0.1,
        allowed_transitions=ext_cfg.allowed_state_transitions,
        allowed_start=ext_cfg.allowed_start_states,
        allowed_end=ext_cfg.allowed_end_states,
        include_start_end_transitions=ext_cfg.use_start_end_transitions,
        constrain_every=True,
    ).to(device)

    T_orig = original_model.crf.transitions.data
    T_ext  = m.crf.transitions.data

    # Copy all original-state transitions verbatim
    T_ext[:N_orig, :N_orig] = T_orig

    # Copy start/end transitions for original states
    m.crf.start_transitions.data[:N_orig] = original_model.crf.start_transitions.data
    m.crf.end_transitions.data[:N_orig]   = original_model.crf.end_transitions.data

    max_mab  = original_cfg.max_segment_lengths['Mab']
    max_mba  = original_cfg.max_segment_lengths['Mba']
    a_alpha  = original_cfg.crf_states['a_alpha']
    b_alpha  = original_cfg.crf_states['b_alpha']
    Mab0     = original_cfg.crf_states['Mab0']
    Mba0     = original_cfg.crf_states['Mba0']
    Mab_last = original_cfg.crf_states[f'Mab{max_mab-1}']
    Mba_last = original_cfg.crf_states[f'Mba{max_mba-1}']
    Raa0     = original_cfg.crf_states['Raa0']
    Rbb0     = original_cfg.crf_states['Rbb0']
    Faa0     = original_cfg.crf_states['Faa0']
    Fbb0     = original_cfg.crf_states['Fbb0']
    a_post   = ext_cfg.crf_states['a_alpha_post']
    b_post   = ext_cfg.crf_states['b_alpha_post']

    # Mab1 internal transitions (mirror Mab)
    for i in range(max_mab):
        for j in range(max_mab):
            T_ext[ext_cfg.crf_states[f'Mab1_{i}'], ext_cfg.crf_states[f'Mab1_{j}']] = \
                T_orig[original_cfg.crf_states[f'Mab{i}'], original_cfg.crf_states[f'Mab{j}']]

    # Mba1 internal transitions (mirror Mba)
    for i in range(max_mba):
        for j in range(max_mba):
            T_ext[ext_cfg.crf_states[f'Mba1_{i}'], ext_cfg.crf_states[f'Mba1_{j}']] = \
                T_orig[original_cfg.crf_states[f'Mba{i}'], original_cfg.crf_states[f'Mba{j}']]

    # Entry into first TM
    T_ext[a_alpha, ext_cfg.crf_states['Mab1_0']] = T_orig[a_alpha, Mab0]
    T_ext[b_alpha, ext_cfg.crf_states['Mba1_0']] = T_orig[b_alpha, Mba0]

    # Exit from Mab1/Mba1 → post states
    T_ext[ext_cfg.crf_states[f'Mab1_{max_mab-1}'], b_post] = T_orig[Mab_last, b_alpha]
    T_ext[ext_cfg.crf_states[f'Mba1_{max_mba-1}'], a_post] = T_orig[Mba_last, a_alpha]

    # Exit from Mab/Mba → post states (second+ TM exits)
    T_ext[Mab_last, b_post] = T_orig[Mab_last, b_alpha]
    T_ext[Mba_last, a_post] = T_orig[Mba_last, a_alpha]

    # Post-state self-loops
    T_ext[a_post, a_post] = T_orig[a_alpha, a_alpha]
    T_ext[b_post, b_post] = T_orig[b_alpha, b_alpha]

    # Second+ TM entries from post states
    T_ext[a_post, Mab0] = T_orig[a_alpha, Mab0]
    T_ext[b_post, Mba0] = T_orig[b_alpha, Mba0]

    # Reentrant / interfacial from post states
    T_ext[a_post, Raa0] = T_orig[a_alpha, Raa0]
    T_ext[b_post, Rbb0] = T_orig[b_alpha, Rbb0]
    T_ext[a_post, Faa0] = T_orig[a_alpha, Faa0]
    T_ext[b_post, Fbb0] = T_orig[b_alpha, Fbb0]

    # End transitions for post states
    m.crf.end_transitions.data[a_post] = original_model.crf.end_transitions.data[a_alpha]
    m.crf.end_transitions.data[b_post] = original_model.crf.end_transitions.data[b_alpha]

    m.crf.do_transition_constraint()

    m.crf_config        = ext_cfg
    m.simplify_labels   = ext_cfg.convert_crf_states_to_labels_batch
    m._tile_emissions   = ext_cfg.tile_emissions

    return m, ext_cfg


def apply_extended_weights(
        original_model,
        original_cfg,
        first_tm_entry=None,
        second_tm_entry=None,
        mab_skip_overrides={},
        mba_skip_overrides={},
        mab1_skip_overrides={},
        mba1_skip_overrides={},
):
    """Build an extended model and optionally override entry / skip weights.

    Parameters
    ----------
    original_model : SP5DeepTMHMM
        Trained model with the original grammar.
    original_cfg : SplitIOCRFwithTPConfig
        CRF config of ``original_model``.
    first_tm_entry : float or None
        If set, overrides T[a_alpha→Mab1_0] and T[b_alpha→Mba1_0].
        ``None`` keeps the trained weight (same cost as the original entry).
    second_tm_entry : float or None
        If set, overrides T[a_alpha_post→Mab0] and T[b_alpha_post→Mba0].
        Negative values penalise second+ TM crossings relative to the first.
    mab_skip_overrides / mba_skip_overrides : dict {seg_len: value}
        Per-length skip-weight overrides for the second+ TM tracks (Mab/Mba).
    mab1_skip_overrides / mba1_skip_overrides : dict {seg_len: value}
        Per-length skip-weight overrides for the first-TM tracks (Mab1/Mba1).

    Returns
    -------
    extended_model : SP5DeepTMHMM
    extended_cfg   : SplitIOCRFwithTPConfig
    """
    m, ext_cfg = make_extended_model(original_model, original_cfg)
    T = m.crf.transitions.data

    if first_tm_entry is not None:
        a = ext_cfg.crf_states['a_alpha']
        b = ext_cfg.crf_states['b_alpha']
        T[a, ext_cfg.crf_states['Mab1_0']] = first_tm_entry
        T[b, ext_cfg.crf_states['Mba1_0']] = first_tm_entry

    if second_tm_entry is not None:
        ap = ext_cfg.crf_states['a_alpha_post']
        bp = ext_cfg.crf_states['b_alpha_post']
        T[ap, ext_cfg.crf_states['Mab0']] = second_tm_entry
        T[bp, ext_cfg.crf_states['Mba0']] = second_tm_entry

    for seg_type, overrides in [('Mab', mab_skip_overrides), ('Mba', mba_skip_overrides)]:
        min_len = original_cfg.min_segment_lengths[seg_type]
        start   = math.ceil(min_len / 2) - 1
        src_idx = ext_cfg.crf_states[seg_type + str(start)]
        for seg_len, val in overrides.items():
            T[src_idx, ext_cfg.crf_states[seg_type + str(seg_len - 1)]] = val

    for seg_type, base, overrides in [('Mab1_', 'Mab', mab1_skip_overrides),
                                       ('Mba1_', 'Mba', mba1_skip_overrides)]:
        min_len = original_cfg.min_segment_lengths[base]
        start   = math.ceil(min_len / 2) - 1
        src_idx = ext_cfg.crf_states[seg_type + str(start)]
        for seg_len, val in overrides.items():
            T[src_idx, ext_cfg.crf_states[seg_type + str(seg_len - 1)]] = val

    m.crf.do_transition_constraint()
    return m, ext_cfg


def get_finetuning_model(original_model, second_tm_entry=-2.0):
    """Return an extended-grammar model ready for fine-tuning.

    This is the recommended entry point for Felix's training code.
    It applies the weight transfer and the empirically determined
    ``second_tm_entry`` penalty that reduces alpha-TM errors by ~13%
    without any retraining.

    Parameters
    ----------
    original_model : SP5DeepTMHMM
        A trained model with the original (non-extended) grammar.
    second_tm_entry : float, optional
        Penalty for the second+ TM entry transition, by default -2.0.
        Larger negative values penalise splitting more aggressively.

    Returns
    -------
    extended_model : SP5DeepTMHMM
        Ready for fine-tuning.  The BiLSTM weights are identical to the
        original; only the CRF state space and transition weights differ.
    """
    return apply_extended_weights(
        original_model,
        original_model.crf_config,
        first_tm_entry=None,
        second_tm_entry=second_tm_entry,
    )[0]

## compute head_pos
def compute_good_coils(raw, t_step=0.01, t_window=0.2, dist_limit=0.005,
                       prefix='', gof_limit=0.98, verbose=None):
    """Compute time-varying coil distances."""
    logger.info('Computing good coil counts')
    try:
        from mne.chpi import compute_chpi_amplitudes, compute_chpi_locs
    except ImportError:
        chpi_locs = _old_chpi_locs(raw, t_step, t_window, prefix)
    else:
        chpi_amps = compute_chpi_amplitudes(
            raw, t_step_min=t_step, t_window=t_window)
        chpi_locs = compute_chpi_locs(raw.info, chpi_amps)
    if len(chpi_locs['rrs']) == 0:
        warnings.warn(
            'No valid cHPI locations found, perhaps cHPI was turned off?')
    from mne.chpi import _get_hpi_initial_fit
    hpi_dig_head_rrs = _get_hpi_initial_fit(raw.info, verbose=False)
    hpi_coil_dists = cdist(hpi_dig_head_rrs, hpi_dig_head_rrs)

    counts = np.empty(len(chpi_locs['times']), int)
    pb = ProgressBar(chpi_locs['gofs'], mesg='Coil distances')
    for ii, (t, coil_dev_rrs, gof) in enumerate(zip(
            chpi_locs['times'], chpi_locs['rrs'], pb)):
        these_dists = cdist(coil_dev_rrs, coil_dev_rrs)
        these_dists = np.abs(hpi_coil_dists - these_dists)
        # there is probably a better algorithm for finding the bad ones...
        use_mask = gof >= gof_limit
        good = False
        while not good:
            d = these_dists[use_mask][:, use_mask]
            d_bad = d > dist_limit
            good = not d_bad.any()
            if not good:
                if use_mask.sum() == 2:
                    use_mask[:] = False
                    break  # failure
                # exclude next worst point
                badness = (d * d_bad).sum(axis=0)
                exclude_coils = np.where(use_mask)[0][np.argmax(badness)]
                use_mask[exclude_coils] = False
        counts[ii] = use_mask.sum()
    t = chpi_locs['times'] - raw.first_samp / raw.info['sfreq']
    logger.info('[done]')
    return t, counts, len(hpi_dig_head_rrs), chpi_locs

fit_t, counts, n_coils, chpi_locs = compute_good_coils(
            raw)
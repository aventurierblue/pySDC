from cupyx.scipy.sparse.linalg import gmres as cupy_gmres


def _effective_restart(operator, restart):
    size = operator.shape[0]
    return min(20 if restart is None else restart, size)


def gmres_with_count(A, b, x0=None, *, restart=None, callback=None, callback_type=None, **kwargs):
    if callback_type not in (None, "pr_norm"):
        raise ValueError("gmres_with_count only supports callback_type=None or 'pr_norm'")

    cycles = 0

    def counting_callback(residual):
        nonlocal cycles
        cycles += 1
        if callback is not None:
            callback(residual)

    x, info = cupy_gmres(
        A,
        b,
        x0=x0,
        restart=restart,
        callback=counting_callback,
        callback_type="pr_norm",
        **kwargs,
    )

    return x, info, cycles * _effective_restart(A, restart)

program test_methods
    use mp_kinds, only : dp
    use mp_linalg, only : linf_norm
    use mp_types, only : method_options, method_stats
    use mp_auzinger, only : auzinger_exact, run_auzinger_newton_sdc, run_auzinger_sdc_newton
    use mp_allencahn, only : allencahn_problem, init_allencahn_problem, run_allencahn_newton_sdc, run_allencahn_sdc_newton
    implicit none

    call test_auzinger_methods()
    call test_allencahn_methods()

contains

    subroutine test_auzinger_methods()
        type(method_options) :: opts
        type(method_stats) :: sdc_stats, newton_stats
        real(dp), allocatable :: sdc_solution(:,:), newton_solution(:,:)
        real(dp) :: err_sdc, err_newton, method_gap

        opts%dt = 5.0e-4_dp
        opts%t_end = 1.0e-2_dp
        opts%outer_tol = 1.0e-10_dp
        opts%inner_tol = 1.0e-10_dp
        opts%outer_maxiter = 24
        opts%inner_maxiter = 6

        call run_auzinger_sdc_newton(3, opts, sdc_solution, sdc_stats)
        call run_auzinger_newton_sdc(3, opts, newton_solution, newton_stats)

        err_sdc = linf_norm(sdc_solution(size(sdc_solution, 1) - 1, :) - auzinger_exact(opts%t_end))
        err_newton = linf_norm(newton_solution(size(newton_solution, 1) - 1, :) - auzinger_exact(opts%t_end))
        method_gap = linf_norm(sdc_solution - newton_solution)

        if (.not. sdc_stats%converged) error stop "Auzinger SDC-Newton did not converge"
        if (.not. newton_stats%converged) error stop "Auzinger Newton-SDC did not converge"
        if (err_sdc > 1.0e-6_dp) error stop "Auzinger SDC-Newton accuracy regression"
        if (err_newton > 1.0e-6_dp) error stop "Auzinger Newton-SDC accuracy regression"
        if (method_gap > 1.0e-8_dp) error stop "Auzinger methods disagree"
    end subroutine test_auzinger_methods

    subroutine test_allencahn_methods()
        type(allencahn_problem) :: problem
        type(method_options) :: opts
        type(method_stats) :: sdc_stats, newton_stats
        real(dp), allocatable :: sdc_state(:,:), newton_state(:,:)
        real(dp) :: method_gap

        call init_allencahn_problem(problem, n=8, nu=2.0_dp, eps=0.04_dp, radius=0.25_dp)

        opts%dt = 5.0e-4_dp
        opts%t_end = 5.0e-4_dp
        opts%outer_tol = 1.0e-8_dp
        opts%inner_tol = 1.0e-8_dp
        opts%linear_tol = 1.0e-10_dp
        opts%outer_maxiter = 20
        opts%inner_maxiter = 4
        opts%linear_maxiter = 100

        call run_allencahn_sdc_newton(problem, 3, opts, sdc_state, sdc_stats)
        call run_allencahn_newton_sdc(problem, 3, opts, newton_state, newton_stats)
        method_gap = linf_norm(sdc_state - newton_state)

        if (.not. sdc_stats%converged) error stop "Allen-Cahn SDC-Newton did not converge"
        if (.not. newton_stats%converged) error stop "Allen-Cahn Newton-SDC did not converge"
        if (method_gap > 1.0e-4_dp) error stop "Allen-Cahn methods diverged too far"
    end subroutine test_allencahn_methods

end program test_methods

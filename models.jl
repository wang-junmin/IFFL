
## load packages
using DifferentialEquations
using DiffEqCallbacks
using PyPlot
using PyCall
using LaTeXStrings
@pyimport matplotlib.gridspec as gspec

########################################################################################################################
###############  ODE Models for IFFLs, Type-1 two input circuits, and negative autoregulated circuits  #################
########################################################################################################################
function i1ffl(du, u, p, t)   # I1-FFL
    b_A, b_B, b_C, a_A, a_B, a_C, K_A, K_B, K_C, n_A, n_B, n_C, eta_A, eta_B, eta_C = p
    A, B, C = abs.(u)
    du[1] = 1/(1+eta_A*n_A^2*(A/K_A)^(n_A-1)/((1+(A/K_A)^n_A)^2)) * (a_A-A)
    du[2] = 1/(1+eta_B*n_B^2*(B/K_B)^(n_B-1)/((1+(B/K_B)^n_B)^2)) * (b_B+(a_B-b_B)*(A^n_A)/(A^n_A+K_A^n_A)-B)
    du[3] = 1/(1+eta_C*n_C^2*(C/K_C)^(n_C-1)/((1+(C/K_C)^n_C)^2)) * (b_C+(a_C-b_C)*(A^n_A)/(A^n_A+K_A^n_A)*(K_B^n_B)/(B^n_B+K_B^n_B)-C)
end

function i4ffl(du, u, p, t)   # I4-FFL
    b_A, b_B, b_C, a_A, a_B, a_C, K_A, K_B, K_C, n_A, n_B, n_C, eta_A, eta_B, eta_C = p
    A, B, C = abs.(u)
    du[1] = 1/(1+eta_A*n_A^2*(A/K_A)^(n_A-1)/((1+(A/K_A)^n_A)^2)) * (a_A-A)
    du[2] = 1/(1+eta_B*n_B^2*(B/K_B)^(n_B-1)/((1+(B/K_B)^n_B)^2)) * (b_B+(a_B-b_B)*(K_A^n_A)/(A^n_A+K_A^n_A)-B)
    du[3] = 1/(1+eta_C*n_C^2*(C/K_C)^(n_C-1)/((1+(C/K_C)^n_C)^2)) * (b_C+(a_C-b_C)*(A^n_A)/(A^n_A+K_A^n_A)*(B^n_B)/(B^n_B+K_B^n_B)-C)
end

function i2ffl(du, u, p, t)   # I2-FFL
    b_A, b_B, b_C, a_A, a_B, a_C, K_A, K_B, K_C, n_A, n_B, n_C, eta_A, eta_B, eta_C = p
    A, B, C = abs.(u)
    du[1] = 1/(1+eta_A*n_A^2*(A/K_A)^(n_A-1)/((1+(A/K_A)^n_A)^2)) * (a_A-A)
    du[2] = 1/(1+eta_B*n_B^2*(B/K_B)^(n_B-1)/((1+(B/K_B)^n_B)^2)) * (b_B+(a_B-b_B)*(K_A^n_A)/(A^n_A+K_A^n_A)-B)
    du[3] = 1/(1+eta_C*n_C^2*(C/K_C)^(n_C-1)/((1+(C/K_C)^n_C)^2)) * (b_C+(a_C-b_C)*(K_AC^n_A)/(A^n_A+K_AC^n_A)*(K_B^n_B)/(B^n_B+K_B^n_B)-C)
end

function i3ffl(du, u, p, t)   # I3-FFL
    b_A, b_B, b_C, a_A, a_B, a_C, K_A, K_B, K_C, n_A, n_B, n_C, eta_A, eta_B, eta_C = p
    A, B, C = abs.(u)
    du[1] = 1/(1+eta_A*n_A^2*(A/K_A)^(n_A-1)/((1+(A/K_A)^n_A)^2)) * (a_A-A)
    du[2] = 1/(1+eta_B*n_B^2*(B/K_B)^(n_B-1)/((1+(B/K_B)^n_B)^2)) * (b_B+(a_B-b_B)*(A^n_A)/(A^n_A+K_A^n_A)-B)
    du[3] = 1/(1+eta_C*n_C^2*(C/K_C)^(n_C-1)/((1+(C/K_C)^n_C)^2)) * (b_C+(a_C-b_C)*(K_A^n_A)/(A^n_A+K_A^n_A)*(B^n_B)/(B^n_B+K_B^n_B)-C)
end

function two_input_circuit_1(du, u, p, t)   # Type-1 two input circuit
    b_A, b_B, b_C, b_A_2, a_A, a_B, a_C, a_A_2, K_A, K_B, K_C, K_A_2, n_A, n_B, n_C, n_A_2, eta_A, eta_B, eta_C, eta_A_2 = p
    A, B, C, A_2 = abs.(u)
    du[1] = 1/(1+eta_A*n_A^2*(A/K_A)^(n_A-1)/((1+(A/K_A)^n_A)^2)) * (a_A-A)
    du[2] = 1/(1+eta_B*n_B^2*(B/K_B)^(n_B-1)/((1+(B/K_B)^n_B)^2)) * (b_B+(a_B-b_B)*(A_2^n_A_2)/(A_2^n_A_2+K_A_2^n_A_2)-B)
    du[3] = 1/(1+eta_C*n_C^2*(C/K_C)^(n_C-1)/((1+(C/K_C)^n_C)^2)) * (b_C+(a_C-b_C)*(A^n_A)/(A^n_A+K_A^n_A)*(K_B^n_B)/(B^n_B+K_B^n_B)-C)
	du[4] = 1/(1+eta_A_2*n_A_2^2*(A_2/K_A_2)^(n_A_2-1)/((1+(A_2/K_A_2)^n_A_2)^2)) * (a_A_2-A_2)
end

function neg_auto(du, u, p, t)   # negative autoregulated circuit
    b_A, b_C, a_A, a_C, K_A, K_C, n_A, n_C, eta_A, eta_C = p
    A, C = abs.(u)
    du[1] = 1/(1+eta_A*n_A^2*(A/K_A)^(n_A-1)/((1+(A/K_A)^n_A)^2)) * (a_A-A)
    du[2] = 1/(1+eta_C*n_C^2*(C/K_C)^(n_C-1)/((1+(C/K_C)^n_C)^2)) * (b_C+(a_C-b_C)*(A^n_A)/(A^n_A+K_A^n_A)*(K_C^n_C)/(K_C^n_C+C^n_C)-C)
end


########################################################################################################################
######  Codes below, which produce Figure 2a, serve to illustrate how to solve ODEs that model IFFLs in Julia  #########
########################################################################################################################
## parameters
H_B_coef = 0.5
H_AC_coef = 1.0
tspan = (0.0, 100000.0)
K_A = 0.1
K_B = 0.1
K_C = 0.1
alpha = 1.0
subfig_ind_lst = [221, 222, 223, 224];
time_right_end = 10.0
eta_B_lst = [0.0, 1.0, 10.0, 100.0];

## create new figure 
PyPlot.matplotlib[:rc]("text", usetex=true)
plt.rc("xtick", labelsize=9)
plt.rc("ytick", labelsize=9)
fig = plt.figure(figsize=(5.0,3.5))
fig.subplots_adjust(left=0.10, right=0.95, top=0.9, bottom=0.25, hspace=0.5)
noretro_sol2_t = 0;
noretro_sol2_A = 0;
big_ax = fig.add_subplot(111);

## Turn off axis lines and ticks
big_ax.spines["top"].set_color("none")
big_ax.spines["bottom"].set_color("none")
big_ax.spines["left"].set_color("none")
big_ax.spines["right"].set_color("none")
big_ax.tick_params(labelcolor="w", top="off", bottom="off", left="off", right="off")

## simulate and plot
for (subfig_ind, eta_B) in zip(subfig_ind_lst, eta_B_lst)
	# pre-induction steady state
	u0 = [1e-5, 1e-5, 1e-5]
	params_pre_ind = [1e-5, 1e-5, 1e-5, 1e-5, alpha, alpha, K_A, K_B, K_C, H_AC_coef, H_B_coef, H_AC_coef, 0.0, eta_B, 0.0]
	cb = TerminateSteadyState()
	prob = ODEProblem(i1ffl, u0, tspan, params_pre_ind)
	sol1 = solve(prob, callback=cb)
	
	# post-induction steady state
	u0 = sol1.u[end]
	params_post_ind = [1e-5, 1e-5, 1e-5, alpha, alpha, alpha, K_A, K_B, K_C, H_AC_coef, H_B_coef, H_AC_coef, 0.0, eta_B, 0.0]
	cb = TerminateSteadyState()
	prob = ODEProblem(i1ffl, u0, tspan, params_post_ind)
	sol2 = solve(prob, callback=cb)
	sol2_A, sol2_B, sol2_C = hcat(sol2.u...)[1,:], hcat(sol2.u...)[2,:], hcat(sol2.u...)[3,:]
	
	# calculate response time
	half_pt = (sol2_C[1] + sol2_C[end]) / 2
	condition(u, t, integrator) = (half_pt-u[3])
	affect!(integrator) = terminate!(integrator)
	cb2 = ContinuousCallback(condition,nothing,affect!)
	sol3 = solve(prob, callback=cb2)
	rt = sol3.t[end]

	if (subfig_ind == 221)
		global noretro_sol2_t = sol2.t;
		global noretro_sol2_C = sol2_C;
	end

	ax = fig.add_subplot(subfig_ind)
	ax.plot(sol2.t, sol2_A, label = latexstring("\$\\tilde{x}_A\$"), color="blue", linewidth=0.5)
	ax.plot(sol2.t, sol2_B, label = latexstring("\$\\tilde{x}_B\$"), color="red", linewidth=0.5)
	ax.plot(sol2.t, sol2_C, label = latexstring("\$\\tilde{x}_C\$"), color="green", linewidth=0.5)
	ax.plot(noretro_sol2_t, noretro_sol2_C, label = latexstring("\$\\tilde{x}_C\$ (No retroactivity)"), "--", color="green", linewidth=0.5)
	ax.set_xlim([0, time_right_end])
	ax.set_title(latexstring("\$\\tilde{\\eta}_{BD_B}=", eta_B, "\$"), fontsize=9)
	handles, labels = ax.get_legend_handles_labels()
	
	if (subfig_ind == 224)
		big_ax.legend(handles, labels, bbox_to_anchor=(0.5, -0.35), ncol=4, fontsize=8, loc="lower center", edgecolor="black")
	end
end

big_ax.set_xlabel("Time (Unit: Mean Lifetime)", fontsize=9);
big_ax.set_ylabel("Protein Concentration", fontsize=9);


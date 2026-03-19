import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from environment import UAVSafeFormationEnv


def run_and_visualize():
    env = UAVSafeFormationEnv(gui=False)
    obs  = env.reset()

    simulation_time = 150.0                              # seconds (matches paper's Fig 1–10 x-axis)
    num_steps       = int(simulation_time / env.controller_dt)

    # ── Data storage ──────────────────────────────────────────────────
    history_pos       = {a: [] for a in env.agents}     # absolute positions (x,y,z)
    history_ei        = {a: [] for a in env.agents}     # position error e_i  (3,)
    history_v         = {a: [] for a in env.agents}     # velocity
    history_psi       = {a: [] for a in env.agents}     # course angle
    history_theta     = {a: [] for a in env.agents}     # pitch angle
    history_vc        = {a: [] for a in env.agents}     # commanded velocity   (control)
    history_psic      = {a: [] for a in env.agents}     # commanded course     (control)
    history_thetac    = {a: [] for a in env.agents}     # commanded pitch      (control)
    history_ki        = {a: [] for a in env.agents}     # safety bound k_ie norm
    history_min_dist  = {a: [] for a in env.agents}     # min distance to neighbour
    history_w_norm    = {a: [] for a in env.agents}     # critic weight norm
    history_leader_pos = []                              # virtual leader position
    times = []
    history_vr = []
    history_psir = []
    history_thetar = []

    print(f"Running simulation for {simulation_time:.0f} seconds "
          f"({num_steps} steps at {1/env.controller_dt:.0f} Hz)...")

    for step_idx in range(num_steps):

        # ── Before step: record control commands from critics ──────────
        # (compute_policy is called inside step(); we mirror it here for logging)
        pre_obs = env._get_obs()
        for agent in env.agents:
            Xi   = pre_obs[agent]
            ei   = Xi[:3]
            k_ie = env._compute_k_ie(agent)
            _, Gi = env._compute_FiGi(agent)
            u_hat, _ = env.critics[agent].compute_policy(
                Xi, ei, k_ie, Gi, env.alpha_i, env.beta_bar_i
            )
            u_hat[0] = np.clip(u_hat[0], env.b_v,     env.h_v)
            u_hat[1] = np.clip(u_hat[1], env.b_psi,   env.h_psi)
            u_hat[2] = np.clip(u_hat[2], env.b_theta, env.h_theta)
            history_vc[agent].append(float(u_hat[0]))
            history_psic[agent].append(float(u_hat[1]))
            history_thetac[agent].append(float(u_hat[2]))

        # ── Step the environment (critics compute policies internally) ─
        obs, reward_dict, done, _, info = env.step()

        # ── Record time and leader ─────────────────────────────────────
        times.append(env.t)
        history_leader_pos.append(env.leader_pos.copy())
        history_vr.append(env.leader_state[0])
        history_psir.append(env.leader_state[1])
        history_thetar.append(env.leader_state[2])

        # ── Record per-agent data ──────────────────────────────────────
        for agent in env.agents:
            Xi    = obs[agent]                              # (6,) augmented state
            ei    = Xi[:3]                                  # position error
            dz    = Xi[3:]                                  # Δζ_i = ζ_i − ζ_r

            # Absolute position from PyBullet
            history_pos[agent].append(env.agent_positions[agent].copy())

            # Formation tracking error
            history_ei[agent].append(ei.copy())

            # Actual flight states ζ_i = Δζ_i + ζ_r
            zeta_i = dz + env.leader_state
            history_v[agent].append(float(zeta_i[0]))
            history_psi[agent].append(float(zeta_i[1]))
            history_theta[agent].append(float(zeta_i[2]))

            # Safety bound norm
            k_ie = env._compute_k_ie(agent)
            history_ki[agent].append(float(np.linalg.norm(k_ie)))

            # Minimum distance to neighbours
            pos_i    = env.agent_positions[agent]
            min_dist = min(
                np.linalg.norm(pos_i - env.agent_positions[other])
                for other in env.agents if other != agent
            )
            history_min_dist[agent].append(min_dist)

            # Critic weight norm
            history_w_norm[agent].append(float(np.linalg.norm(env.critics[agent].w)))

        if done:
            print(f"Formation achieved at t = {env.t:.2f} s  (step {step_idx})")
            break

        if step_idx % 1000 == 0:
            print(f"  t = {env.t:7.2f} s   |   step {step_idx}/{num_steps}")

    env.close()

    # ── Convert to numpy ───────────────────────────────────────────────
    times              = np.array(times)
    history_leader_pos = np.array(history_leader_pos)

    for agent in env.agents:
        history_pos[agent]      = np.array(history_pos[agent])
        history_ei[agent]       = np.array(history_ei[agent])
        history_v[agent]        = np.array(history_v[agent])
        history_psi[agent]      = np.array(history_psi[agent])
        history_theta[agent]    = np.array(history_theta[agent])
        history_vc[agent]       = np.array(history_vc[agent])
        history_psic[agent]     = np.array(history_psic[agent])
        history_thetac[agent]   = np.array(history_thetac[agent])
        history_ki[agent]       = np.array(history_ki[agent])
        history_min_dist[agent] = np.array(history_min_dist[agent])
        history_w_norm[agent]   = np.array(history_w_norm[agent])
        history_vr = np.array(history_vr)
        history_psir = np.array(history_psir)
        history_thetar = np.array(history_thetar)

    print("Simulation complete. Generating plots...")
    plot_results(
        env.agents, times,
        history_pos, history_leader_pos,
        history_ei,
        history_v, history_psi, history_theta,
        history_vc, history_psic, history_thetac,
        history_ki, history_min_dist,
        history_w_norm,
        history_vr, history_psir, history_thetar,
        env
    )


# ======================================================================
# PLOTTING  —  mirrors Figs 1–10 from the paper
# ======================================================================

def plot_results(agents, times,
                 history_pos, history_leader_pos,
                 history_ei,
                 history_v, history_psi, history_theta,
                 history_vc, history_psic, history_thetac,
                 history_ki, history_min_dist,
                 history_w_norm,
                 history_vr, history_psir, history_thetar,
                 env):

    plt.style.use('seaborn-v0_8-whitegrid')
    colors      = ['tab:blue', 'tab:green', 'tab:red', 'tab:cyan', 'tab:orange']
    linestyles  = ['-', '--', '-.', ':', (0, (3, 1, 1, 1))]
    n_agents    = len(agents)

    # leader reference values (constant in Section VI)
    vr    = env.leader_state[0]
    psi_r = env.leader_state[1]
    thr   = env.leader_state[2]

    # ── Fig 1: Velocity tracking  ──────────────────────────────────────
    fig1, ax1 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents):
        ax1.plot(times, history_v[agent],
                 color=colors[i], ls=linestyles[i],
                 label=f'$v_{i+1}$', linewidth=1.5)
    ax1.plot(times, history_vr, 'k--', linewidth=1.2, label='$v_r$')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Velocity $v_i$ (m/s)')
    ax1.set_title('Fig 1 — Velocity tracking $v_i$ vs $v_r$')
    ax1.legend(ncol=3)
    fig1.tight_layout()

    # ── Fig 2: Course angle tracking  ─────────────────────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents):
        ax2.plot(times, history_psi[agent], color=colors[i], ls=linestyles[i], label=f'$\\psi_{i+1}$', linewidth=1.5)
    ax2.plot(times, history_psir, 'k--', linewidth=1.2, label='$\\psi_r$')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Course angle $\\psi_i$ (deg)')
    ax2.set_title('Fig 2 — Course angle tracking $\\psi_i$ vs $\\psi_r$')
    ax2.legend(ncol=3)
    fig2.tight_layout()

    # ── Fig 3: Pitch angle tracking  ──────────────────────────────────
    fig3, ax3 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents):
        ax3.plot(times, history_theta[agent], color=colors[i], ls=linestyles[i], label=f'$\\theta_{i + 1}$',
                 linewidth=1.5)
    ax3.plot(times, history_thetar, 'k--', linewidth=1.2, label='$\\theta_r$')
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Pitch angle $\\theta_i$ (deg)')
    ax3.set_title('Fig 3 — Pitch angle tracking $\\theta_i$ vs $\\theta_r$')
    ax3.legend(ncol=3)
    fig3.tight_layout()

    # ── Fig 4: Commanded velocity v^c_i  ──────────────────────────────
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents):
        ax4.plot(times, history_vc[agent],
                 color=colors[i], ls=linestyles[i],
                 label=f'$v^c_{i+1}$', linewidth=1.5)
    ax4.axhline(env.b_v, color='k', ls='--', linewidth=1.0, label='Constraints')
    ax4.axhline(env.h_v, color='k', ls='--', linewidth=1.0)
    ax4.set_xlabel('Time (s)')
    ax4.set_ylabel('$v^c_i$ (m/s)')
    ax4.set_title('Fig 4 — Commanded velocity $v^c_i$ with constraints')
    ax4.legend(ncol=3)
    fig4.tight_layout()

    # ── Fig 5: Commanded course angle ψ^c_i  ──────────────────────────
    fig5, ax5 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents):
        ax5.plot(times, history_psic[agent],
                 color=colors[i], ls=linestyles[i],
                 label=f'$\\psi^c_{i+1}$', linewidth=1.5)
    ax5.axhline(env.b_psi, color='k', ls='--', linewidth=1.0, label='Constraints')
    ax5.axhline(env.h_psi, color='k', ls='--', linewidth=1.0)
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('$\\psi^c_i$ (deg)')
    ax5.set_title('Fig 5 — Commanded course angle $\\psi^c_i$ with constraints')
    ax5.legend(ncol=3)
    fig5.tight_layout()

    # ── Fig 6: Commanded pitch angle θ^c_i  ───────────────────────────
    fig6, ax6 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents):
        ax6.plot(times, history_thetac[agent],
                 color=colors[i], ls=linestyles[i],
                 label=f'$\\theta^c_{i+1}$', linewidth=1.5)
    ax6.axhline(env.b_theta, color='k', ls='--', linewidth=1.0, label='Constraints')
    ax6.axhline(env.h_theta, color='k', ls='--', linewidth=1.0)
    ax6.set_xlabel('Time (s)')
    ax6.set_ylabel('$\\theta^c_i$ (deg)')
    ax6.set_title('Fig 6 — Commanded pitch angle $\\theta^c_i$ with constraints')
    ax6.legend(ncol=3)
    fig6.tight_layout()

    # ── Fig 7: 3D Formation Trajectory  ───────────────────────────────
    fig7  = plt.figure(figsize=(12, 8))
    ax_3d = fig7.add_subplot(111, projection='3d')

    for i, agent in enumerate(agents):
        pos = history_pos[agent]
        ax_3d.plot(pos[:, 0], pos[:, 1], pos[:, 2],
                   color=colors[i], ls=linestyles[i],
                   label=f'UAV$_{i+1}$', linewidth=1.5)
        # Start marker
        ax_3d.scatter(pos[0,  0], pos[0,  1], pos[0,  2],
                      marker='o', s=60,  color=colors[i], zorder=5)
        # End marker
        ax_3d.scatter(pos[-1, 0], pos[-1, 1], pos[-1, 2],
                      marker='*', s=120, color=colors[i], zorder=5)

    # Leader trajectory
    ax_3d.plot(history_leader_pos[:, 0],
               history_leader_pos[:, 1],
               history_leader_pos[:, 2],
               'k--', linewidth=1.5, label='Leader')
    ax_3d.scatter(*history_leader_pos[0],  marker='D', s=80,  color='k')
    ax_3d.scatter(*history_leader_pos[-1], marker='P', s=120, color='k')

    ax_3d.set_xlabel('x (m)')
    ax_3d.set_ylabel('y (m)')
    ax_3d.set_zlabel('z (m)')
    ax_3d.set_title('Fig 7 — 3D Formation Trajectory')
    ax_3d.legend()
    fig7.tight_layout()

    # ── Fig 8: Tracking error norm vs safety bound k_ie  ──────────────
    fig8, axs8 = plt.subplots(n_agents, 1, figsize=(10, 3 * n_agents), sharex=True)
    if n_agents == 1:
        axs8 = [axs8]

    for i, agent in enumerate(agents):
        ei_norm = np.linalg.norm(history_ei[agent], axis=1)
        axs8[i].plot(times, ei_norm,
                     color=colors[i], linewidth=1.5,
                     label=f'$\\|e_{i+1}\\|$')
        axs8[i].plot(times, history_ki[agent],
                     color=colors[i], ls='--', linewidth=1.2,
                     label=f'$k_{{i{i+1}}}$')
        axs8[i].set_ylabel(f'UAV$_{i+1}$ error / bound (m)')
        axs8[i].legend(loc='upper right')

    axs8[-1].set_xlabel('Time (s)')
    fig8.suptitle('Fig 8 — Tracking error norm $\\|e_i\\|$ vs safety bound $k_i$')
    fig8.tight_layout()

    # ── Fig 9: Minimum inter-UAV distance  ────────────────────────────
    fig9, ax9 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents):
        ax9.plot(times, history_min_dist[agent],
                 color=colors[i], ls=linestyles[i],
                 label=f'min dist UAV$_{i+1}$', linewidth=1.5)
    ax9.set_xlabel('Time (s)')
    ax9.set_ylabel('Minimum distance to neighbour (m)')
    ax9.set_title('Fig 9 — Minimum inter-UAV distances (collision avoidance)')
    ax9.legend(ncol=2)
    fig9.tight_layout()

    # ── Fig 10: Critic weight norms  ──────────────────────────────────
    fig10, ax10 = plt.subplots(figsize=(10, 4))
    for i, agent in enumerate(agents):
        ax10.plot(times, history_w_norm[agent],
                  color=colors[i], ls=linestyles[i],
                  label=f'$\\|\\hat{{w}}_{i+1}\\|$', linewidth=1.5)
    ax10.set_xlabel('Time (s)')
    ax10.set_ylabel('Critic weight norm $\\|\\hat{w}_i\\|$')
    ax10.set_title('Fig 10 — Critic weight convergence')
    ax10.legend(ncol=3)
    fig10.tight_layout()

    plt.show()


if __name__ == "__main__":
    run_and_visualize()
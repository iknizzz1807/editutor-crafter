<script lang="ts">
	import ProgressCircle from './ProgressCircle.svelte';

	let {
		domains,
		totalMilestones,
		totalCompleted,
		user,
		currentDomainSlug,
		open
	}: {
		domains: Array<{
			id: number;
			slug: string;
			name: string;
			icon: string | null;
			totalMilestones: number;
			completedMilestones: number;
		}>;
		totalMilestones: number;
		totalCompleted: number;
		user: { id: number; username: string; email: string } | null;
		currentDomainSlug: string;
		open: boolean;
	} = $props();

	let overallPct = $derived(totalMilestones > 0 ? (totalCompleted / totalMilestones) * 100 : 0);

	function projectCount(domain: (typeof domains)[0]) {
		return domain.totalMilestones;
	}
</script>

<aside class="sidebar" class:open>
	<div class="sidebar-header">
		<div class="logo">
			<span class="logo-icon">🎯</span>
			EduTutor Crafter
		</div>
	</div>

	<div class="progress-overview">
		<div class="progress-label">Overall Progress</div>
		<div class="progress-bar">
			<div class="progress-fill" style="width: {overallPct}%"></div>
		</div>
		<div class="progress-text">
			<strong>{totalCompleted}</strong> of {totalMilestones} milestones completed
		</div>
	</div>

	<nav class="domain-nav">
		<div class="nav-section-title">Domains</div>
		{#each domains as domain}
			<a
				href="/roadmap/{domain.slug}"
				class="domain-item"
				class:active={currentDomainSlug === domain.slug}
			>
				<span class="domain-icon">{domain.icon || '📁'}</span>
				<div class="domain-info">
					<div class="domain-name">{domain.name}</div>
					<div class="domain-count">{domain.totalMilestones} milestones</div>
				</div>
				<ProgressCircle
					completed={domain.completedMilestones}
					total={domain.totalMilestones}
					size={32}
				/>
			</a>
		{/each}
	</nav>

	{#if user}
		<div class="sidebar-footer">
			<div class="user-info">
				<div class="user-avatar">{user.username[0].toUpperCase()}</div>
				<span class="user-name">{user.username}</span>
			</div>
			<form method="POST" action="/auth/logout">
				<button type="submit" class="logout-btn">Logout</button>
			</form>
		</div>
	{/if}
</aside>

<style>
	.sidebar {
		width: 280px;
		background: var(--bg-sidebar);
		border-right: 1px solid var(--border);
		height: 100vh;
		position: fixed;
		left: 0;
		top: 0;
		overflow-y: auto;
		z-index: 100;
		transition: transform 0.3s ease;
		display: flex;
		flex-direction: column;
	}

	.sidebar-header {
		padding: 20px;
		border-bottom: 1px solid var(--border);
	}

	.logo {
		display: flex;
		align-items: center;
		gap: 12px;
		font-weight: 700;
		font-size: 18px;
		color: var(--text-primary);
	}

	.logo-icon {
		width: 32px;
		height: 32px;
		background: linear-gradient(135deg, var(--accent-bright), var(--blue));
		border-radius: 8px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 16px;
	}

	.progress-overview {
		padding: 16px 20px;
		border-bottom: 1px solid var(--border);
	}

	.progress-label {
		font-size: 12px;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 8px;
	}

	.progress-bar {
		height: 8px;
		background: var(--bg-card);
		border-radius: 4px;
		overflow: hidden;
	}

	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--accent-bright), var(--blue));
		border-radius: 4px;
		transition: width 0.5s ease;
	}

	.progress-text {
		font-size: 13px;
		color: var(--text-secondary);
		margin-top: 8px;
	}

	.progress-text strong {
		color: var(--accent-bright);
	}

	.domain-nav {
		padding: 12px 0;
		flex: 1;
	}

	.nav-section-title {
		padding: 8px 20px;
		font-size: 11px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}

	.domain-item {
		display: flex;
		align-items: center;
		gap: 12px;
		padding: 10px 20px;
		cursor: pointer;
		transition: all 0.15s ease;
		border-left: 2px solid transparent;
		text-decoration: none;
		color: var(--text-primary);
	}

	.domain-item:hover {
		background: var(--bg-card);
		text-decoration: none;
	}

	.domain-item.active {
		background: var(--bg-card);
		border-left-color: var(--accent-bright);
	}

	.domain-icon {
		font-size: 18px;
		width: 24px;
		text-align: center;
	}

	.domain-info {
		flex: 1;
		min-width: 0;
	}

	.domain-name {
		font-size: 14px;
		font-weight: 500;
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.domain-count {
		font-size: 12px;
		color: var(--text-muted);
	}

	.sidebar-footer {
		padding: 16px 20px;
		border-top: 1px solid var(--border);
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.user-info {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.user-avatar {
		width: 28px;
		height: 28px;
		background: var(--accent);
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 12px;
		font-weight: 700;
		color: white;
	}

	.user-name {
		font-size: 13px;
		color: var(--text-secondary);
	}

	.logout-btn {
		background: none;
		border: 1px solid var(--border);
		border-radius: 4px;
		padding: 4px 10px;
		font-size: 12px;
		color: var(--text-muted);
		cursor: pointer;
		font-family: inherit;
	}

	.logout-btn:hover {
		color: var(--red);
		border-color: var(--red);
	}

	@media (max-width: 768px) {
		.sidebar {
			transform: translateX(-100%);
		}

		.sidebar.open {
			transform: translateX(0);
		}
	}
</style>

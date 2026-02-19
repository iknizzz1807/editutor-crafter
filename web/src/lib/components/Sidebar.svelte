<script lang="ts">
	import ProgressCircle from './ProgressCircle.svelte';
	import DomainIcon from './DomainIcon.svelte';
	import { sidebarCollapsed } from '$lib/stores/sidebar.js';

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

	let collapsed = $state(false);

	sidebarCollapsed.subscribe((v) => (collapsed = v));

	function toggleCollapse() {
		collapsed = !collapsed;
		sidebarCollapsed.set(collapsed);
	}

	let overallPct = $derived(totalMilestones > 0 ? (totalCompleted / totalMilestones) * 100 : 0);
</script>

<aside class="sidebar" class:open class:collapsed>
	<div class="sidebar-header">
		<a href="/roadmap" class="logo">
			<span class="logo-icon">🎯</span>
			{#if !collapsed}
				<span class="logo-text">EduTutor Crafter</span>
			{/if}
		</a>
		<button class="toggle-btn" onclick={toggleCollapse} title={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}>
			<span class="toggle-icon" class:collapsed>{collapsed ? '▶' : '◀'}</span>
		</button>
	</div>

	{#if !collapsed}
		<div class="progress-overview">
			<div class="progress-label">Overall Progress</div>
			<div class="progress-bar">
				<div class="progress-fill" style="width: {overallPct}%"></div>
			</div>
			<div class="progress-text">
				<strong>{totalCompleted}</strong> of {totalMilestones} milestones completed
			</div>
		</div>
	{:else}
		<div class="progress-overview-mini">
			<ProgressCircle completed={totalCompleted} total={totalMilestones} size={36} />
		</div>
	{/if}

	<nav class="domain-nav">
		{#if !collapsed}
			<div class="nav-section-title">Domains</div>
		{/if}
		{#each domains as domain}
			<a
				href="/roadmap/{domain.slug}"
				class="domain-item"
				class:active={currentDomainSlug === domain.slug}
				title={collapsed ? domain.name : ''}
			>
				<span class="domain-icon"><DomainIcon icon={domain.icon} /></span>
				{#if !collapsed}
					<div class="domain-info">
						<div class="domain-name">{domain.name}</div>
						<div class="domain-count">{domain.totalMilestones} milestones</div>
					</div>
					<ProgressCircle
						completed={domain.completedMilestones}
						total={domain.totalMilestones}
						size={32}
					/>
				{/if}
			</a>
		{/each}
	</nav>

	{#if user}
		<nav class="extra-nav">
			<a href="/history" class="extra-nav-item" title={collapsed ? 'History' : ''}>
				<span class="extra-nav-icon">🕐</span>
				{#if !collapsed}<span>History</span>{/if}
			</a>
			<a href="/settings" class="extra-nav-item" title={collapsed ? 'Settings' : ''}>
				<span class="extra-nav-icon">⚙️</span>
				{#if !collapsed}<span>Settings</span>{/if}
			</a>
		</nav>
		<div class="sidebar-footer">
			<div class="user-info">
				<div class="user-avatar">{user.username[0].toUpperCase()}</div>
				{#if !collapsed}
					<span class="user-name">{user.username}</span>
				{/if}
			</div>
			{#if !collapsed}
				<form method="POST" action="/auth/logout">
					<button type="submit" class="logout-btn">Logout</button>
				</form>
			{/if}
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
		transition: width 0.25s ease, transform 0.3s ease;
		display: flex;
		flex-direction: column;
	}

	.sidebar.collapsed {
		width: 60px;
	}

	.sidebar-header {
		padding: 16px;
		border-bottom: 1px solid var(--border);
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 8px;
		min-height: 64px;
	}

	.logo {
		display: flex;
		align-items: center;
		gap: 12px;
		font-weight: 700;
		font-size: 18px;
		color: var(--text-primary);
		text-decoration: none;
		min-width: 0;
		overflow: hidden;
	}

	.logo:hover {
		text-decoration: none;
	}

	.logo-text {
		white-space: nowrap;
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
		flex-shrink: 0;
	}

	.toggle-btn {
		background: none;
		border: none;
		color: var(--text-muted);
		cursor: pointer;
		padding: 4px;
		border-radius: 4px;
		display: flex;
		align-items: center;
		justify-content: center;
		flex-shrink: 0;
		font-size: 12px;
		transition: color 0.15s;
	}

	.toggle-btn:hover {
		color: var(--text-primary);
		background: var(--bg-card);
	}

	.collapsed .toggle-btn {
		margin: 0 auto;
	}

	.progress-overview {
		padding: 16px 20px;
		border-bottom: 1px solid var(--border);
	}

	.progress-overview-mini {
		padding: 12px 0;
		border-bottom: 1px solid var(--border);
		display: flex;
		justify-content: center;
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

	.collapsed .domain-item {
		justify-content: center;
		padding: 10px 0;
		border-left: none;
	}

	.domain-item:hover {
		background: var(--bg-card);
		text-decoration: none;
	}

	.domain-item.active {
		background: var(--bg-card);
		border-left-color: var(--accent-bright);
	}

	.collapsed .domain-item.active {
		border-left-color: transparent;
		position: relative;
	}

	.collapsed .domain-item.active::before {
		content: '';
		position: absolute;
		left: 0;
		top: 50%;
		transform: translateY(-50%);
		width: 3px;
		height: 20px;
		background: var(--accent-bright);
		border-radius: 0 2px 2px 0;
	}

	.domain-icon {
		font-size: 18px;
		width: 24px;
		text-align: center;
		flex-shrink: 0;
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

	.extra-nav {
		padding: 8px 0;
		border-top: 1px solid var(--border);
	}

	.extra-nav-item {
		display: flex;
		align-items: center;
		gap: 10px;
		padding: 8px 20px;
		font-size: 13px;
		color: var(--text-secondary);
		text-decoration: none;
		transition: all 0.15s ease;
	}

	.collapsed .extra-nav-item {
		justify-content: center;
		padding: 8px 0;
	}

	.extra-nav-item:hover {
		background: var(--bg-card);
		color: var(--text-primary);
		text-decoration: none;
	}

	.extra-nav-icon {
		width: 20px;
		text-align: center;
		font-size: 14px;
		flex-shrink: 0;
	}

	.sidebar-footer {
		padding: 16px 20px;
		border-top: 1px solid var(--border);
		display: flex;
		align-items: center;
		justify-content: space-between;
	}

	.collapsed .sidebar-footer {
		padding: 12px 0;
		justify-content: center;
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
		flex-shrink: 0;
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
			width: 280px !important;
			transform: translateX(-100%);
		}

		.sidebar.open {
			transform: translateX(0);
		}

		.toggle-btn {
			display: none;
		}
	}
</style>

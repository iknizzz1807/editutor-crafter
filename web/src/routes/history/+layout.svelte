<script lang="ts">
	import Sidebar from '$lib/components/Sidebar.svelte';
	import { sidebarCollapsed } from '$lib/stores/sidebar.js';

	let { children, data } = $props();
	let sidebarOpen = $state(false);
	let collapsed = $state(false);

	sidebarCollapsed.subscribe((v) => (collapsed = v));
</script>

<div class="app-layout">
	<div class="mobile-header">
		<button class="menu-btn" onclick={() => (sidebarOpen = true)}>&#9776;</button>
		<span class="logo">
			<span class="logo-icon">🎯</span>
			EduTutor Crafter
		</span>
	</div>

	{#if sidebarOpen}
		<!-- svelte-ignore a11y_click_events_have_key_events -->
		<!-- svelte-ignore a11y_no_static_element_interactions -->
		<div class="sidebar-overlay" onclick={() => (sidebarOpen = false)}></div>
	{/if}

	<Sidebar
		domains={data.domains}
		totalMilestones={data.totalMilestones}
		totalCompleted={data.totalCompleted}
		user={data.user}
		currentDomainSlug=""
		open={sidebarOpen}
	/>

	<main class="main" class:collapsed>
		{@render children()}
	</main>
</div>

<style>
	.app-layout {
		display: flex;
		min-height: 100vh;
	}

	.main {
		flex: 1;
		margin-left: 280px;
		min-height: 100vh;
		transition: margin-left 0.25s ease;
	}

	.main.collapsed {
		margin-left: 60px;
	}

	.mobile-header {
		display: none;
		padding: 16px 20px;
		background: var(--bg-sidebar);
		border-bottom: 1px solid var(--border);
		position: sticky;
		top: 0;
		z-index: 200;
		align-items: center;
		gap: 16px;
	}

	.menu-btn {
		width: 40px;
		height: 40px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 8px;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 20px;
		color: var(--text-primary);
	}

	.logo {
		display: flex;
		align-items: center;
		gap: 10px;
		font-weight: 700;
		font-size: 18px;
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

	.sidebar-overlay {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		bottom: 0;
		background: rgba(0, 0, 0, 0.5);
		z-index: 99;
	}

	@media (max-width: 768px) {
		.mobile-header {
			display: flex;
		}

		.main {
			margin-left: 0 !important;
		}
	}
</style>

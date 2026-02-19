<script lang="ts">
	import ProjectCard from '$lib/components/ProjectCard.svelte';
	import DomainIcon from '$lib/components/DomainIcon.svelte';

	let { data } = $props();

	let currentFilter = $state('all');

	let levels = $derived(
		currentFilter === 'all'
			? (['beginner', 'intermediate', 'advanced', 'expert'] as const)
			: ([currentFilter] as const)
	);

	const levelLabels: Record<string, string> = {
		beginner: 'Beginner',
		intermediate: 'Intermediate',
		advanced: 'Advanced',
		expert: 'Expert'
	};
</script>

<svelte:head>
	<title>{data.domain.name} - EduTutor Crafter</title>
</svelte:head>

<header class="main-header">
	<div class="header-top">
		<div class="header-left">
			<h1>
				<span class="domain-icon"><DomainIcon icon={data.domain.icon} /></span>
				{data.domain.name}
			</h1>
			{#if data.domain.description}
				<p>{data.domain.description}</p>
			{/if}
		</div>

		<div class="filters">
			<button
				class="filter-btn"
				class:active={currentFilter === 'all'}
				onclick={() => (currentFilter = 'all')}
			>
				All Levels
			</button>
			{#each ['beginner', 'intermediate', 'advanced', 'expert'] as level}
				<button
					class="filter-btn"
					class:active={currentFilter === level}
					onclick={() => (currentFilter = level)}
				>
					{levelLabels[level]}
				</button>
			{/each}
		</div>
	</div>
</header>

<div class="content">
	{#each levels as level}
		{@const levelProjects = data.projects[level as keyof typeof data.projects] || []}
		{#if levelProjects.length > 0}
			<div class="level-group">
				<div class="level-header">
					<span class="level-badge {level}">{levelLabels[level]}</span>
					<span class="level-count">{levelProjects.length} projects</span>
				</div>
				<div class="projects-grid">
					{#each levelProjects as project}
						<ProjectCard {project} domainSlug={data.domain.slug} />
					{/each}
				</div>
			</div>
		{/if}
	{/each}
</div>

<style>
	.main-header {
		padding: 24px 32px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-sidebar);
		position: sticky;
		top: 0;
		z-index: 50;
	}

	.main-header h1 {
		font-size: 24px;
		font-weight: 700;
		display: flex;
		align-items: center;
		gap: 12px;
		margin-bottom: 4px;
	}

	.domain-icon {
		display: flex;
		align-items: center;
		justify-content: center;
	}

	.main-header p {
		color: var(--text-secondary);
		margin-top: 0;
		font-size: 14px;
		margin-bottom: 16px;
	}

	.header-top {
		display: flex;
		justify-content: space-between;
		align-items: flex-start;
		gap: 20px;
	}

	.header-left {
		flex: 1;
		min-width: 0;
	}

	.filters {
		display: flex;
		gap: 8px;
		flex-wrap: wrap;
		flex-shrink: 0;
	}

	.filter-btn {
		padding: 6px 12px;
		border: 1px solid var(--border);
		background: transparent;
		border-radius: 6px;
		font-size: 13px;
		color: var(--text-secondary);
		cursor: pointer;
		transition: all 0.15s ease;
		font-family: inherit;
	}

	.filter-btn:hover {
		border-color: var(--text-muted);
		color: var(--text-primary);
	}

	.filter-btn.active {
		background: var(--accent);
		border-color: var(--accent);
		color: white;
	}

	.content {
		padding: 24px 32px;
	}

	.level-group {
		margin-bottom: 32px;
	}

	.level-header {
		display: flex;
		align-items: center;
		gap: 12px;
		margin-bottom: 16px;
		padding-bottom: 12px;
		border-bottom: 1px solid var(--border);
	}

	.level-badge {
		padding: 4px 10px;
		border-radius: 4px;
		font-size: 12px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}

	.level-badge.beginner {
		background: rgba(63, 185, 80, 0.15);
		color: var(--beginner);
	}
	.level-badge.intermediate {
		background: rgba(210, 153, 34, 0.15);
		color: var(--intermediate);
	}
	.level-badge.advanced {
		background: rgba(240, 136, 62, 0.15);
		color: var(--advanced);
	}
	.level-badge.expert {
		background: rgba(248, 81, 73, 0.15);
		color: var(--expert);
	}

	.level-count {
		font-size: 13px;
		color: var(--text-muted);
	}

	.projects-grid {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	@media (max-width: 768px) {
		.content {
			padding: 16px;
		}

		.main-header {
			display: block;
			padding: 16px;
		}
	}
</style>

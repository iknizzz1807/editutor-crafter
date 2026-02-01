<script lang="ts">
	let {
		project,
		domainSlug
	}: {
		project: {
			id: number;
			slug: string;
			name: string;
			description: string | null;
			difficulty: string;
			estimatedHours: string | null;
			totalMilestones: number;
			completedMilestones: number;
		};
		domainSlug: string;
	} = $props();

	let progressPct = $derived(
		project.totalMilestones > 0
			? (project.completedMilestones / project.totalMilestones) * 100
			: 0
	);
</script>

<a href="/roadmap/{domainSlug}/{project.slug}" class="project-card">
	<div class="project-header">
		<div class="project-info">
			<div class="project-title">
				{project.name}
			</div>
			{#if project.description}
				<div class="project-desc">{project.description}</div>
			{/if}
		</div>
		<div class="project-meta">
			{#if project.totalMilestones > 0}
				<span class="meta-item">📋 {project.completedMilestones}/{project.totalMilestones}</span>
			{/if}
			{#if project.estimatedHours}
				<span class="meta-item">⏱️ {project.estimatedHours}h</span>
			{/if}
		</div>
	</div>
	{#if project.totalMilestones > 0}
		<div class="project-progress-bar">
			<div class="project-progress-fill" style="width: {progressPct}%"></div>
		</div>
	{/if}
</a>

<style>
	.project-card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 8px;
		overflow: hidden;
		transition: all 0.2s ease;
		text-decoration: none;
		color: var(--text-primary);
		display: block;
	}

	.project-card:hover {
		border-color: var(--text-muted);
		text-decoration: none;
	}

	.project-header {
		padding: 16px 20px;
		display: flex;
		align-items: center;
		gap: 16px;
	}

	.project-info {
		flex: 1;
		min-width: 0;
	}

	.project-title {
		font-size: 15px;
		font-weight: 600;
		margin-bottom: 4px;
	}

	.project-desc {
		font-size: 13px;
		color: var(--text-secondary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
	}

	.project-meta {
		display: flex;
		align-items: center;
		gap: 12px;
		flex-shrink: 0;
	}

	.meta-item {
		font-size: 12px;
		color: var(--text-muted);
	}

	.project-progress-bar {
		height: 3px;
		background: var(--bg-dark);
	}

	.project-progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--accent-bright), var(--blue));
		transition: width 0.5s ease;
	}
</style>

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

	const difficultyColors: Record<string, string> = {
		beginner: 'var(--beginner)',
		intermediate: 'var(--intermediate)',
		advanced: 'var(--advanced)',
		expert: 'var(--expert)'
	};

	let diffColor = $derived(difficultyColors[project.difficulty] || 'var(--text-muted)');
</script>

<a href="/roadmap/{domainSlug}/{project.slug}" class="project-card">
	<div class="card-top">
		<div class="project-info">
			<div class="title-row">
				<span class="project-title">{project.name}</span>
				<span class="difficulty-badge" style="--diff-color: {diffColor}">
					{project.difficulty}
				</span>
			</div>
			{#if project.description}
				<div class="project-desc">{project.description}</div>
			{/if}
		</div>
	</div>

	<div class="card-bottom">
		<div class="progress-section">
			{#if project.totalMilestones > 0}
				<div class="progress-info">
					<span class="progress-label">
						<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
							<path d="M8 9.5a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"/>
							<path fill-rule="evenodd" d="M8 0a8 8 0 100 16A8 8 0 008 0zM1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0z"/>
						</svg>
						{project.completedMilestones}/{project.totalMilestones} milestones
					</span>
					<span class="progress-pct">{Math.round(progressPct)}%</span>
				</div>
				<div class="progress-bar">
					<div
						class="progress-fill"
						class:complete={progressPct === 100}
						style="width: {progressPct}%"
					></div>
				</div>
			{/if}
		</div>
		{#if project.estimatedHours}
			<span class="hours-badge">
				<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
					<path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zm.5 4.75a.75.75 0 00-1.5 0v3.5a.75.75 0 00.37.65l2.5 1.5a.75.75 0 10.76-1.3L8.5 7.96V4.75z"/>
				</svg>
				{project.estimatedHours}h
			</span>
		{/if}
	</div>
</a>

<style>
	.project-card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
		transition: all 0.2s ease;
		text-decoration: none;
		color: var(--text-primary);
		display: flex;
		flex-direction: column;
		min-width: 0;
	}

	.project-card:hover {
		border-color: var(--text-muted);
		box-shadow: var(--shadow-md);
		text-decoration: none;
		transform: translateY(-1px);
	}

	.card-top {
		padding: 16px 20px 12px;
		min-width: 0;
	}

	.project-info {
		min-width: 0;
	}

	.title-row {
		display: flex;
		align-items: center;
		gap: 10px;
		margin-bottom: 4px;
	}

	.project-title {
		font-size: 15px;
		font-weight: 600;
		flex: 1;
		min-width: 0;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.difficulty-badge {
		padding: 2px 8px;
		border-radius: 12px;
		font-size: 11px;
		font-weight: 500;
		text-transform: capitalize;
		background: color-mix(in srgb, var(--diff-color) 15%, transparent);
		color: var(--diff-color);
		flex-shrink: 0;
		letter-spacing: 0.3px;
	}

	.project-desc {
		font-size: 13px;
		color: var(--text-secondary);
		white-space: nowrap;
		overflow: hidden;
		text-overflow: ellipsis;
		line-height: 1.5;
	}

	.card-bottom {
		padding: 0 20px 14px;
		display: flex;
		align-items: flex-end;
		gap: 12px;
		margin-top: auto;
	}

	.progress-section {
		flex: 1;
		min-width: 0;
	}

	.progress-info {
		display: flex;
		align-items: center;
		justify-content: space-between;
		margin-bottom: 6px;
	}

	.progress-label {
		font-size: 12px;
		color: var(--text-muted);
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.progress-pct {
		font-size: 11px;
		color: var(--text-muted);
		font-weight: 500;
	}

	.progress-bar {
		height: 6px;
		background: var(--bg-dark);
		border-radius: 3px;
		overflow: hidden;
	}

	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--accent-bright), var(--blue));
		border-radius: 3px;
		transition: width 0.5s ease;
	}

	.progress-fill.complete {
		background: var(--accent-bright);
	}

	.hours-badge {
		display: flex;
		align-items: center;
		gap: 4px;
		font-size: 12px;
		color: var(--text-muted);
		flex-shrink: 0;
		padding-bottom: 1px;
	}
</style>

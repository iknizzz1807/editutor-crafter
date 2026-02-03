<script lang="ts">
	import { enhance } from '$app/forms';
	import MilestoneTimeline from '$lib/components/MilestoneTimeline.svelte';

	let { data } = $props();

	let completedCount = $derived(
		data.milestones.filter((m) => m.status === 'completed').length
	);
	let totalCount = $derived(data.milestones.length);
	let progressPct = $derived(totalCount > 0 ? (completedCount / totalCount) * 100 : 0);

	const difficultyColors: Record<string, string> = {
		beginner: 'var(--beginner)',
		intermediate: 'var(--intermediate)',
		advanced: 'var(--advanced)',
		expert: 'var(--expert)'
	};

	let diffColor = $derived(difficultyColors[data.project.difficulty] || 'var(--text-muted)');
</script>

<svelte:head>
	<title>{data.project.name} - EduTutor Crafter</title>
</svelte:head>

<header class="main-header">
	<div class="breadcrumb">
		<a href="/roadmap">Roadmap</a>
		<span class="sep">/</span>
		<a href="/roadmap/{data.domain.slug}">{data.domain.icon} {data.domain.name}</a>
		<span class="sep">/</span>
		<span class="current">{data.project.name}</span>
	</div>

	<div class="header-top">
		<div class="header-info">
			<h1>{data.project.name}</h1>
			{#if data.project.description}
				<p class="project-desc">{data.project.description}</p>
			{/if}
		</div>
		<span class="difficulty-pill" style="--diff-color: {diffColor}">
			{data.project.difficulty}
		</span>
	</div>

	<div class="header-progress">
		<div class="progress-row">
			<div class="progress-stats">
				<span class="stat-chip">
					<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
						<path d="M8 9.5a1.5 1.5 0 100-3 1.5 1.5 0 000 3z"/>
						<path fill-rule="evenodd" d="M8 0a8 8 0 100 16A8 8 0 008 0zM1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0z"/>
					</svg>
					{completedCount}/{totalCount} milestones
				</span>
				{#if data.project.estimatedHours}
					<span class="stat-chip">
						<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
							<path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zm.5 4.75a.75.75 0 00-1.5 0v3.5a.75.75 0 00.37.65l2.5 1.5a.75.75 0 10.76-1.3L8.5 7.96V4.75z"/>
						</svg>
						{data.project.estimatedHours} hours
					</span>
				{/if}
				<span class="stat-chip pct">{Math.round(progressPct)}% complete</span>
			</div>
		</div>
		{#if totalCount > 0}
			<div class="progress-bar">
				<div
					class="progress-fill"
					class:complete={progressPct === 100}
					style="width: {progressPct}%"
				></div>
			</div>
		{/if}
	</div>
</header>

<div class="content">
	{#if data.project.essence}
		<p class="project-essence">{data.project.essence}</p>
	{/if}

	{#if data.project.whyImportant}
		<div class="info-section">
			<h3>
				<svg width="16" height="16" viewBox="0 0 16 16" fill="var(--orange)">
					<path fill-rule="evenodd" d="M8 1.5c-2.363 0-4 1.69-4 3.75 0 .984.424 1.625.984 2.304l.214.253c.223.264.47.556.673.848.284.411.537.896.621 1.49a.75.75 0 01-1.484.211c-.04-.282-.163-.547-.37-.847a8.695 8.695 0 00-.542-.68c-.084-.1-.173-.205-.268-.32C3.201 7.75 2.5 6.766 2.5 5.25 2.5 2.31 4.863 0 8 0s5.5 2.31 5.5 5.25c0 1.516-.701 2.5-1.328 3.259-.095.115-.184.22-.268.319-.207.245-.383.453-.541.681-.208.3-.33.565-.37.847a.75.75 0 01-1.485-.212c.084-.593.337-1.078.621-1.489.203-.292.45-.584.673-.848l.213-.253c.561-.679.985-1.32.985-2.304 0-2.06-1.637-3.75-4-3.75zM6 15.25a.75.75 0 01.75-.75h2.5a.75.75 0 010 1.5h-2.5a.75.75 0 01-.75-.75zM5.75 12a.75.75 0 000 1.5h4.5a.75.75 0 000-1.5h-4.5z"/>
				</svg>
				Why This Project?
			</h3>
			<p class="info-text">{data.project.whyImportant}</p>
		</div>
	{/if}

	{#if data.outcomes.length > 0}
		<div class="info-section">
			<h3>
				<svg width="16" height="16" viewBox="0 0 16 16" fill="var(--accent-bright)">
					<path fill-rule="evenodd" d="M13.78 4.22a.75.75 0 010 1.06l-7.25 7.25a.75.75 0 01-1.06 0L2.22 9.28a.75.75 0 011.06-1.06L6 10.94l6.72-6.72a.75.75 0 011.06 0z"/>
				</svg>
				Learning Outcomes
			</h3>
			<ul class="outcomes-list">
				{#each data.outcomes as outcome}
					<li>{outcome.outcome}</li>
				{/each}
			</ul>
		</div>
	{/if}

	{#if data.prerequisites.length > 0}
		<div class="info-section">
			<h3>
				<svg width="16" height="16" viewBox="0 0 16 16" fill="var(--red)">
					<path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"/>
				</svg>
				Prerequisites
			</h3>
			<div class="tag-list">
				{#each data.prerequisites as prereq}
					<span class="prereq-tag">{prereq.prerequisiteName}</span>
				{/each}
			</div>
		</div>
	{/if}

	{#if data.languages.length > 0}
		<div class="info-section">
			<h3>
				<svg width="16" height="16" viewBox="0 0 16 16" fill="var(--blue)">
					<path fill-rule="evenodd" d="M4.72 3.22a.75.75 0 011.06 0l3.25 3.25a.75.75 0 010 1.06L5.78 10.78a.75.75 0 01-1.06-1.06L7.44 7 4.72 4.28a.75.75 0 010-1.06zm4.25 7.5a.75.75 0 000 1.5h2.75a.75.75 0 000-1.5H8.97z"/>
				</svg>
				Languages
			</h3>
			<div class="tag-list">
				{#each data.languages as lang}
					<span class="lang-tag" class:recommended={lang.recommended === 1}>
						{lang.language}
						{#if lang.recommended === 1}
							<span class="rec-dot"></span>
						{/if}
					</span>
				{/each}
			</div>
		</div>
	{/if}

	{#if data.resources.length > 0}
		<div class="info-section">
			<h3>
				<svg width="16" height="16" viewBox="0 0 16 16" fill="var(--purple)">
					<path fill-rule="evenodd" d="M0 1.75A.75.75 0 01.75 1h4.253c1.227 0 2.317.59 3 1.501A3.744 3.744 0 0111.006 1h4.245a.75.75 0 01.75.75v10.5a.75.75 0 01-.75.75h-4.507a2.25 2.25 0 00-1.591.659l-.622.621a.75.75 0 01-1.06 0l-.622-.621A2.25 2.25 0 005.258 13H.75a.75.75 0 01-.75-.75V1.75zm7.251 10.324l.004-5.073-.002-2.253A2.25 2.25 0 005.003 2.5H1.5v9h3.757a3.75 3.75 0 011.994.574zM8.755 4.75l-.004 7.322a3.752 3.752 0 011.992-.572H14.5v-9h-3.495a2.25 2.25 0 00-2.25 2.25z"/>
				</svg>
				Resources
			</h3>
			<ul class="resources-list">
				{#each data.resources as resource}
					<li>
						<a href={resource.url} target="_blank" rel="noopener">
							<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
								<path fill-rule="evenodd" d="M10.604 1h4.146a.25.25 0 01.25.25v4.146a.25.25 0 01-.427.177L13.03 4.03 9.28 7.78a.75.75 0 01-1.06-1.06l3.75-3.75-1.543-1.543A.25.25 0 0110.604 1zM3.75 2A1.75 1.75 0 002 3.75v8.5c0 .966.784 1.75 1.75 1.75h8.5A1.75 1.75 0 0014 12.25v-3.5a.75.75 0 00-1.5 0v3.5a.25.25 0 01-.25.25h-8.5a.25.25 0 01-.25-.25v-8.5a.25.25 0 01.25-.25h3.5a.75.75 0 000-1.5h-3.5z"/>
							</svg>
							{resource.title}
						</a>
						{#if resource.resourceType}
							<span class="resource-type">{resource.resourceType}</span>
						{/if}
					</li>
				{/each}
			</ul>
		</div>
	{/if}

	{#if data.tags.length > 0}
		<div class="info-section">
			<h3>
				<svg width="16" height="16" viewBox="0 0 16 16" fill="var(--text-muted)">
					<path fill-rule="evenodd" d="M2.5 7.775V2.75a.25.25 0 01.25-.25h5.025a.25.25 0 01.177.073l6.25 6.25a.25.25 0 010 .354l-5.025 5.025a.25.25 0 01-.354 0l-6.25-6.25a.25.25 0 01-.073-.177zm-1.5 0V2.75C1 1.784 1.784 1 2.75 1h5.025c.464 0 .91.184 1.238.513l6.25 6.25a1.75 1.75 0 010 2.474l-5.026 5.026a1.75 1.75 0 01-2.474 0l-6.25-6.25A1.75 1.75 0 011 7.775zM6 5a1 1 0 100 2 1 1 0 000-2z"/>
				</svg>
				Tags
			</h3>
			<div class="tag-list">
				{#each data.tags as tag}
					<span class="tag-item">{tag.tag}</span>
				{/each}
			</div>
		</div>
	{/if}

	{#if data.milestones.length > 0}
		<div class="milestones-section">
			<h3>
				<svg width="18" height="18" viewBox="0 0 16 16" fill="var(--blue)">
					<path fill-rule="evenodd" d="M8 .25a.75.75 0 01.673.418l1.882 3.815 4.21.612a.75.75 0 01.416 1.279l-3.046 2.97.719 4.192a.75.75 0 01-1.088.791L8 12.347l-3.766 1.98a.75.75 0 01-1.088-.79l.72-4.194L.818 6.374a.75.75 0 01.416-1.28l4.21-.611L7.327.668A.75.75 0 018 .25z"/>
				</svg>
				Milestones
			</h3>
			<MilestoneTimeline milestones={data.milestones} hasApiKey={data.hasApiKey} />
		</div>
	{/if}
</div>

<style>
	.main-header {
		padding: 24px 32px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-sidebar);
	}

	.breadcrumb {
		font-size: 13px;
		color: var(--text-muted);
		margin-bottom: 16px;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.breadcrumb a {
		color: var(--text-secondary);
		text-decoration: none;
	}

	.breadcrumb a:hover {
		color: var(--blue);
		text-decoration: none;
	}

	.sep {
		color: var(--border);
	}

	.current {
		color: var(--text-primary);
		font-weight: 500;
	}

	.header-top {
		display: flex;
		align-items: flex-start;
		gap: 16px;
		margin-bottom: 16px;
	}

	.header-info {
		flex: 1;
		min-width: 0;
	}

	.main-header h1 {
		font-size: 24px;
		font-weight: 700;
	}

	.project-desc {
		color: var(--text-secondary);
		margin-top: 4px;
		font-size: 14px;
	}

	.difficulty-pill {
		padding: 4px 14px;
		border-radius: 20px;
		font-size: 12px;
		font-weight: 600;
		text-transform: capitalize;
		background: color-mix(in srgb, var(--diff-color) 15%, transparent);
		color: var(--diff-color);
		flex-shrink: 0;
		letter-spacing: 0.3px;
		margin-top: 4px;
	}

	.header-progress {
		margin-top: 0;
	}

	.progress-row {
		margin-bottom: 8px;
	}

	.progress-stats {
		display: flex;
		align-items: center;
		gap: 10px;
		flex-wrap: wrap;
	}

	.stat-chip {
		display: flex;
		align-items: center;
		gap: 5px;
		font-size: 13px;
		color: var(--text-secondary);
	}

	.stat-chip.pct {
		margin-left: auto;
		font-weight: 600;
		color: var(--text-primary);
	}

	.progress-bar {
		height: 8px;
		background: var(--bg-elevated);
		border-radius: 4px;
		overflow: hidden;
	}

	.progress-fill {
		height: 100%;
		background: linear-gradient(90deg, var(--accent-bright), var(--blue));
		border-radius: 4px;
		transition: width 0.5s ease;
	}

	.progress-fill.complete {
		background: var(--accent-bright);
	}

	.content {
		padding: 24px 32px;
	}

	.project-essence {
		font-size: 14px;
		font-style: italic;
		color: var(--text-muted);
		line-height: 1.6;
		margin-bottom: 24px;
		padding: 12px 16px;
		border-left: 3px solid var(--purple);
		background: rgba(163, 113, 247, 0.05);
		border-radius: 0 6px 6px 0;
	}

	.info-section {
		margin-bottom: 24px;
	}

	.info-section h3 {
		font-size: 14px;
		font-weight: 600;
		margin-bottom: 12px;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.info-text {
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.6;
	}

	.outcomes-list {
		list-style: none;
	}

	.outcomes-list li {
		position: relative;
		padding: 5px 0 5px 22px;
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.5;
	}

	.outcomes-list li::before {
		content: '✦';
		position: absolute;
		left: 0;
		color: var(--accent-bright);
	}

	.tag-list {
		display: flex;
		flex-wrap: wrap;
		gap: 6px;
	}

	.prereq-tag {
		padding: 4px 10px;
		background: rgba(248, 81, 73, 0.1);
		border: 1px solid rgba(248, 81, 73, 0.2);
		border-radius: 20px;
		font-size: 12px;
		color: var(--red);
	}

	.lang-tag {
		padding: 4px 10px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 20px;
		font-size: 12px;
		color: var(--text-muted);
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.lang-tag.recommended {
		background: rgba(88, 166, 255, 0.1);
		border-color: rgba(88, 166, 255, 0.3);
		color: var(--blue);
	}

	.rec-dot {
		width: 6px;
		height: 6px;
		border-radius: 50%;
		background: var(--blue);
	}

	.resources-list {
		list-style: none;
	}

	.resources-list li {
		padding: 6px 0;
		font-size: 13px;
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}

	.resources-list a {
		color: var(--blue);
		text-decoration: none;
		display: flex;
		align-items: center;
		gap: 5px;
	}

	.resources-list a:hover {
		text-decoration: underline;
	}

	.resource-type {
		font-size: 11px;
		padding: 2px 8px;
		background: var(--bg-card);
		border-radius: 12px;
		color: var(--text-muted);
	}

	.tag-item {
		padding: 3px 10px;
		background: var(--bg-card);
		border-radius: 12px;
		font-size: 11px;
		color: var(--text-muted);
	}

	.milestones-section {
		margin-top: 32px;
	}

	.milestones-section h3 {
		font-size: 16px;
		font-weight: 600;
		margin-bottom: 20px;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	@media (max-width: 768px) {
		.main-header {
			padding: 16px;
		}

		.content {
			padding: 16px;
		}

		.header-top {
			flex-direction: column;
			gap: 8px;
		}

		.stat-chip.pct {
			margin-left: 0;
		}
	}
</style>

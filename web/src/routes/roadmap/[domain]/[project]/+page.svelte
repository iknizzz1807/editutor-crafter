<script lang="ts">
	import { enhance } from '$app/forms';
	import MilestoneTimeline from '$lib/components/MilestoneTimeline.svelte';

	let { data } = $props();

	let completedCount = $derived(
		data.milestones.filter((m) => m.status === 'completed').length
	);
	let totalCount = $derived(data.milestones.length);
	let progressPct = $derived(totalCount > 0 ? (completedCount / totalCount) * 100 : 0);
</script>

<svelte:head>
	<title>{data.project.name} - EduTutor Crafter</title>
</svelte:head>

<header class="main-header">
	<div class="breadcrumb">
		<a href="/roadmap/{data.domain.slug}">{data.domain.icon} {data.domain.name}</a>
		<span class="separator">/</span>
		<span>{data.project.name}</span>
	</div>
	<h1>{data.project.name}</h1>
	{#if data.project.description}
		<p class="project-desc">{data.project.description}</p>
	{/if}

	<div class="detail-stats">
		{#if data.project.estimatedHours}
			<div class="stat-item">
				<div class="stat-icon">⏱️</div>
				<div class="stat-info">
					<strong>{data.project.estimatedHours} hours</strong>
					<span>Estimated time</span>
				</div>
			</div>
		{/if}
		<div class="stat-item">
			<div class="stat-icon">📋</div>
			<div class="stat-info">
				<strong>{completedCount}/{totalCount} milestones</strong>
				<span>{Math.round(progressPct)}% complete</span>
			</div>
		</div>
		<div class="stat-item">
			<div class="stat-icon">📊</div>
			<div class="stat-info">
				<strong class="difficulty-{data.project.difficulty}">{data.project.difficulty}</strong>
				<span>Difficulty</span>
			</div>
		</div>
	</div>
</header>

<div class="content">
	{#if data.project.essence}
		<p class="project-essence">{data.project.essence}</p>
	{/if}

	{#if data.project.whyImportant}
		<div class="info-section">
			<h3>💡 Why This Project?</h3>
			<p class="info-text">{data.project.whyImportant}</p>
		</div>
	{/if}

	{#if data.outcomes.length > 0}
		<div class="info-section">
			<h3>🎓 Learning Outcomes</h3>
			<ul class="outcomes-list">
				{#each data.outcomes as outcome}
					<li>{outcome.outcome}</li>
				{/each}
			</ul>
		</div>
	{/if}

	{#if data.prerequisites.length > 0}
		<div class="info-section">
			<h3>📝 Prerequisites</h3>
			<div class="tag-list">
				{#each data.prerequisites as prereq}
					<span class="prereq-tag">{prereq.prerequisiteName}</span>
				{/each}
			</div>
		</div>
	{/if}

	{#if data.languages.length > 0}
		<div class="info-section">
			<h3>💻 Languages</h3>
			<div class="tag-list">
				{#each data.languages as lang}
					<span class="lang-tag" class:recommended={lang.recommended === 1}>
						{lang.language}
					</span>
				{/each}
			</div>
		</div>
	{/if}

	{#if data.resources.length > 0}
		<div class="info-section">
			<h3>📖 Resources</h3>
			<ul class="resources-list">
				{#each data.resources as resource}
					<li>
						<a href={resource.url} target="_blank" rel="noopener">{resource.title}</a>
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
			<h3>🏷️ Tags</h3>
			<div class="tag-list">
				{#each data.tags as tag}
					<span class="tag-item">{tag.tag}</span>
				{/each}
			</div>
		</div>
	{/if}

	{#if data.milestones.length > 0}
		<div class="milestones-section">
			<h3>📍 Milestones</h3>
			<MilestoneTimeline milestones={data.milestones} languages={data.languages} hasApiKey={data.hasApiKey} />
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
		margin-bottom: 12px;
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.breadcrumb a {
		color: var(--text-secondary);
		text-decoration: none;
	}

	.breadcrumb a:hover {
		color: var(--text-primary);
	}

	.separator {
		color: var(--border);
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

	.detail-stats {
		display: flex;
		gap: 24px;
		margin-top: 16px;
	}

	.stat-item {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.stat-icon {
		width: 32px;
		height: 32px;
		background: var(--bg-card);
		border-radius: 6px;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 14px;
	}

	.stat-info {
		font-size: 13px;
	}

	.stat-info strong {
		display: block;
		color: var(--text-primary);
		text-transform: capitalize;
	}

	.stat-info span {
		color: var(--text-muted);
		font-size: 11px;
	}

	:global(.difficulty-beginner) {
		color: var(--beginner) !important;
	}
	:global(.difficulty-intermediate) {
		color: var(--intermediate) !important;
	}
	:global(.difficulty-advanced) {
		color: var(--advanced) !important;
	}
	:global(.difficulty-expert) {
		color: var(--expert) !important;
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
		padding: 4px 0 4px 20px;
		font-size: 13px;
		color: var(--text-secondary);
	}

	.outcomes-list li::before {
		content: '✦';
		position: absolute;
		left: 0;
		color: var(--purple);
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
		border-radius: 4px;
		font-size: 12px;
		color: var(--red);
	}

	.lang-tag {
		padding: 4px 10px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 4px;
		font-size: 12px;
		color: var(--text-muted);
	}

	.lang-tag.recommended {
		background: rgba(88, 166, 255, 0.1);
		border-color: rgba(88, 166, 255, 0.3);
		color: var(--blue);
	}

	.resources-list {
		list-style: none;
	}

	.resources-list li {
		padding: 6px 0;
		font-size: 13px;
	}

	.resources-list a {
		color: var(--blue);
		text-decoration: none;
	}

	.resources-list a:hover {
		text-decoration: underline;
	}

	.resource-type {
		font-size: 11px;
		padding: 2px 6px;
		background: var(--bg-card);
		border-radius: 3px;
		color: var(--text-muted);
		margin-left: 6px;
	}

	.tag-item {
		padding: 3px 8px;
		background: var(--bg-card);
		border-radius: 3px;
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
	}

	@media (max-width: 768px) {
		.main-header {
			padding: 16px;
		}

		.content {
			padding: 16px;
		}

		.detail-stats {
			flex-wrap: wrap;
			gap: 12px;
		}
	}
</style>

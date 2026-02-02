<script lang="ts">
	let { data } = $props();

	let expandedIds = $state(new Set<number>());

	function toggleExpand(id: number) {
		if (expandedIds.has(id)) {
			expandedIds.delete(id);
		} else {
			expandedIds.add(id);
		}
		expandedIds = new Set(expandedIds);
	}

	function formatDate(iso: string) {
		return new Date(iso).toLocaleDateString('en-US', {
			year: 'numeric',
			month: 'short',
			day: 'numeric',
			hour: '2-digit',
			minute: '2-digit'
		});
	}
</script>

<svelte:head>
	<title>History - EduTutor Crafter</title>
</svelte:head>

<header class="page-header">
	<h1>Submission History</h1>
	<p class="page-desc">Your past submissions and AI reviews</p>
</header>

<div class="history-content">
	{#if data.submissions.length === 0}
		<div class="empty-state">
			<div class="empty-icon">📝</div>
			<h3>No submissions yet</h3>
			<p>Submit your work on a milestone to see it here</p>
		</div>
	{:else}
		<div class="submissions-list">
			{#each data.submissions as submission}
				{@const isExpanded = expandedIds.has(submission.id)}
				<div class="submission-card">
					<button class="submission-header" onclick={() => toggleExpand(submission.id)}>
						<div class="submission-info">
							<div class="submission-meta">
								<span class="domain-badge">{submission.domainIcon || '📁'} {submission.domainName}</span>
								<span class="separator">/</span>
								<span class="project-name">{submission.projectName}</span>
							</div>
							<div class="submission-title">{submission.milestoneTitle}</div>
							<div class="submission-details">
								<span class="submission-date">{formatDate(submission.createdAt)}</span>
								{#if submission.language}
									<span class="lang-badge">{submission.language}</span>
								{/if}
								<span class="status-badge" class:reviewed={submission.status === 'reviewed'}>
									{submission.status}
								</span>
								{#if submission.review}
									<span class="review-badge">AI Reviewed</span>
								{/if}
							</div>
						</div>
						<span class="toggle-icon" class:open={isExpanded}>▼</span>
					</button>

					{#if isExpanded}
						<div class="submission-body">
							<div class="code-section">
								<div class="section-title">Submission</div>
								<pre class="code-block"><code>{submission.content}</code></pre>
							</div>

							{#if submission.review}
								<div class="review-section">
									<div class="section-title">AI Review</div>
									<div class="review-content">{submission.review}</div>
								</div>
							{/if}
						</div>
					{/if}
				</div>
			{/each}
		</div>
	{/if}
</div>

<style>
	.page-header {
		padding: 24px 32px;
		border-bottom: 1px solid var(--border);
		background: var(--bg-sidebar);
	}

	.page-header h1 {
		font-size: 24px;
		font-weight: 700;
	}

	.page-desc {
		color: var(--text-secondary);
		font-size: 14px;
		margin-top: 4px;
	}

	.history-content {
		padding: 24px 32px;
	}

	.empty-state {
		text-align: center;
		padding: 60px 20px;
		color: var(--text-muted);
	}

	.empty-icon {
		font-size: 48px;
		margin-bottom: 16px;
	}

	.empty-state h3 {
		font-size: 18px;
		color: var(--text-secondary);
		margin-bottom: 8px;
	}

	.empty-state p {
		font-size: 14px;
	}

	.submissions-list {
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.submission-card {
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 8px;
		overflow: hidden;
	}

	.submission-header {
		width: 100%;
		padding: 14px 16px;
		display: flex;
		align-items: flex-start;
		justify-content: space-between;
		gap: 12px;
		background: none;
		border: none;
		color: var(--text-primary);
		font-family: inherit;
		text-align: left;
		cursor: pointer;
	}

	.submission-header:hover {
		background: var(--bg-card-hover);
	}

	.submission-info {
		flex: 1;
		min-width: 0;
	}

	.submission-meta {
		display: flex;
		align-items: center;
		gap: 6px;
		font-size: 12px;
		color: var(--text-muted);
		margin-bottom: 4px;
	}

	.separator {
		color: var(--border);
	}

	.project-name {
		color: var(--text-secondary);
	}

	.domain-badge {
		color: var(--text-secondary);
	}

	.submission-title {
		font-size: 14px;
		font-weight: 600;
		margin-bottom: 6px;
	}

	.submission-details {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-wrap: wrap;
	}

	.submission-date {
		font-size: 12px;
		color: var(--text-muted);
	}

	.lang-badge {
		padding: 1px 6px;
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 3px;
		font-size: 11px;
		color: var(--text-muted);
	}

	.status-badge {
		padding: 1px 6px;
		background: rgba(210, 153, 34, 0.15);
		color: var(--orange);
		border-radius: 3px;
		font-size: 11px;
		text-transform: capitalize;
	}

	.status-badge.reviewed {
		background: rgba(63, 185, 80, 0.15);
		color: var(--accent-bright);
	}

	.review-badge {
		padding: 1px 6px;
		background: rgba(163, 113, 247, 0.15);
		color: var(--purple);
		border-radius: 3px;
		font-size: 11px;
	}

	.toggle-icon {
		color: var(--text-muted);
		font-size: 12px;
		flex-shrink: 0;
		margin-top: 4px;
		transition: transform 0.2s ease;
	}

	.toggle-icon.open {
		transform: rotate(180deg);
	}

	.submission-body {
		border-top: 1px solid var(--border);
		padding: 16px;
	}

	.section-title {
		font-size: 12px;
		font-weight: 600;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 10px;
	}

	.code-block {
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: 6px;
		padding: 14px;
		overflow-x: auto;
		font-family: 'JetBrains Mono', monospace;
		font-size: 13px;
		line-height: 1.5;
		color: var(--text-secondary);
		white-space: pre-wrap;
		word-break: break-word;
	}

	.review-section {
		margin-top: 20px;
	}

	.review-content {
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-left: 3px solid var(--purple);
		border-radius: 0 6px 6px 0;
		padding: 14px;
		font-size: 13px;
		line-height: 1.6;
		color: var(--text-secondary);
		white-space: pre-wrap;
		word-break: break-word;
	}

	@media (max-width: 768px) {
		.page-header {
			padding: 16px;
		}
		.history-content {
			padding: 16px;
		}
	}
</style>

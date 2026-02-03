<script lang="ts">
	import FileTreeBrowser from '$lib/components/FileTreeBrowser.svelte';

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

	function formatSize(bytes: number): string {
		if (bytes < 1024) return bytes + ' B';
		if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
		return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
	}

	const statusConfig: Record<string, { color: string; bg: string; border: string }> = {
		reviewed: {
			color: 'var(--accent-bright)',
			bg: 'rgba(63, 185, 80, 0.08)',
			border: 'rgba(63, 185, 80, 0.3)'
		},
		pending: {
			color: 'var(--orange)',
			bg: 'rgba(210, 153, 34, 0.08)',
			border: 'rgba(210, 153, 34, 0.3)'
		}
	};

	function getStatusStyle(status: string) {
		return statusConfig[status] || statusConfig['pending'];
	}
</script>

<svelte:head>
	<title>History - EduTutor Crafter</title>
</svelte:head>

<header class="page-header">
	<h1>
		<svg width="22" height="22" viewBox="0 0 16 16" fill="var(--blue)">
			<path fill-rule="evenodd" d="M1.643 3.143L.427 1.927A.25.25 0 000 2.104V5.75c0 .138.112.25.25.25h3.646a.25.25 0 00.177-.427L2.715 4.215a6.5 6.5 0 11-1.18 4.458.75.75 0 10-1.493.154 8.001 8.001 0 101.6-5.684zM7.75 4a.75.75 0 01.75.75v2.992l2.028.812a.75.75 0 01-.557 1.392l-2.5-1A.75.75 0 017 8.25v-3.5A.75.75 0 017.75 4z"/>
		</svg>
		Submission History
	</h1>
	<p class="page-desc">Your past submissions and AI reviews</p>
</header>

<div class="history-content">
	{#if data.submissions.length === 0}
		<div class="empty-state">
			<div class="empty-icon">
				<svg width="48" height="48" viewBox="0 0 16 16" fill="var(--text-muted)">
					<path fill-rule="evenodd" d="M0 1.75C0 .784.784 0 1.75 0h12.5C15.216 0 16 .784 16 1.75v12.5A1.75 1.75 0 0114.25 16H1.75A1.75 1.75 0 010 14.25V1.75zM1.5 1.75v12.5c0 .138.112.25.25.25h12.5a.25.25 0 00.25-.25V1.75a.25.25 0 00-.25-.25H1.75a.25.25 0 00-.25.25zM5 8.75a.75.75 0 01.75-.75h4.5a.75.75 0 010 1.5h-4.5A.75.75 0 015 8.75zm0 2.5a.75.75 0 01.75-.75h4.5a.75.75 0 010 1.5h-4.5a.75.75 0 01-.75-.75z"/>
				</svg>
			</div>
			<h3>No submissions yet</h3>
			<p>Submit your work on a milestone to see it here</p>
		</div>
	{:else}
		<div class="submissions-list">
			{#each data.submissions as submission}
				{@const isExpanded = expandedIds.has(submission.id)}
				{@const sStyle = getStatusStyle(submission.status)}
				<div
					class="submission-card"
					class:expanded={isExpanded}
					style="--card-accent: {sStyle.border}"
				>
					<button class="submission-header" onclick={() => toggleExpand(submission.id)}>
						<div class="submission-info">
							<div class="submission-meta">
								<span class="domain-badge">{submission.domainIcon || '📁'} {submission.domainName}</span>
								<span class="meta-sep">/</span>
								<span class="project-name">{submission.projectName}</span>
							</div>
							<div class="submission-title">{submission.milestoneTitle}</div>
							<div class="submission-details">
								<span class="submission-date">
									<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
										<path fill-rule="evenodd" d="M1.5 8a6.5 6.5 0 1113 0 6.5 6.5 0 01-13 0zM8 0a8 8 0 100 16A8 8 0 008 0zm.5 4.75a.75.75 0 00-1.5 0v3.5a.75.75 0 00.37.65l2.5 1.5a.75.75 0 10.76-1.3L8.5 7.96V4.75z"/>
									</svg>
									{formatDate(submission.createdAt)}
								</span>
								<span class="file-badge">
									<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
										<path fill-rule="evenodd" d="M3.5 1.75a.25.25 0 01.25-.25h3.168a.25.25 0 01.177.073l4.832 4.832a.25.25 0 010 .354l-3.168 3.168a.25.25 0 01-.354 0L3.573 4.927a.25.25 0 01-.073-.177V1.75z"/>
									</svg>
									{submission.fileName}
									<span class="file-size">({formatSize(submission.fileSize)})</span>
								</span>
								<span
									class="status-badge"
									style="color: {sStyle.color}; background: {sStyle.bg}"
								>
									{submission.status}
								</span>
								{#if submission.review}
									<span class="review-badge">
										<svg width="10" height="10" viewBox="0 0 16 16" fill="currentColor">
											<path fill-rule="evenodd" d="M1.5 2.75a.25.25 0 01.25-.25h8.5a.25.25 0 01.25.25v5.5a.25.25 0 01-.25.25h-3.5a.75.75 0 00-.53.22L3.5 11.44V9.25a.75.75 0 00-.75-.75h-.5a.25.25 0 01-.25-.25v-5.5zM1.75 1A1.75 1.75 0 000 2.75v5.5C0 9.216.784 10 1.75 10H2v1.543a1.457 1.457 0 002.487 1.03L7.061 10h3.189A1.75 1.75 0 0012 8.25v-5.5A1.75 1.75 0 0010.25 1h-8.5zM14.5 4.75a.25.25 0 00-.25-.25h-.5a.75.75 0 010-1.5h.5c.966 0 1.75.784 1.75 1.75v5.5A1.75 1.75 0 0114.25 12H14v1.543a1.457 1.457 0 01-2.487 1.03L9.22 12.28a.75.75 0 011.06-1.06l2.22 2.22v-2.19a.75.75 0 01.75-.75h1a.25.25 0 00.25-.25v-5.5z"/>
										</svg>
										AI Reviewed
									</span>
								{/if}
							</div>
						</div>
						<span class="toggle-icon" class:open={isExpanded}>
							<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
								<path fill-rule="evenodd" d="M12.78 6.22a.75.75 0 010 1.06l-4.25 4.25a.75.75 0 01-1.06 0L3.22 7.28a.75.75 0 011.06-1.06L8 9.94l3.72-3.72a.75.75 0 011.06 0z"/>
							</svg>
						</span>
					</button>

					{#if isExpanded}
						<div class="submission-body">
							<FileTreeBrowser submissionId={submission.id} fileName={submission.fileName} />

							{#if submission.review}
								<div class="review-section">
									<div class="section-title">
										<svg width="14" height="14" viewBox="0 0 16 16" fill="var(--purple)">
											<path fill-rule="evenodd" d="M1.5 2.75a.25.25 0 01.25-.25h8.5a.25.25 0 01.25.25v5.5a.25.25 0 01-.25.25h-3.5a.75.75 0 00-.53.22L3.5 11.44V9.25a.75.75 0 00-.75-.75h-.5a.25.25 0 01-.25-.25v-5.5zM1.75 1A1.75 1.75 0 000 2.75v5.5C0 9.216.784 10 1.75 10H2v1.543a1.457 1.457 0 002.487 1.03L7.061 10h3.189A1.75 1.75 0 0012 8.25v-5.5A1.75 1.75 0 0010.25 1h-8.5z"/>
										</svg>
										AI Review
									</div>
									<div class="review-content">{submission.review}</div>
								</div>
							{:else}
								<p class="no-review">No AI review yet. Request one from the project page.</p>
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
		display: flex;
		align-items: center;
		gap: 10px;
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
		border-left: 3px solid var(--card-accent, var(--border));
		border-radius: var(--radius-md);
		overflow: hidden;
		transition: border-color 0.2s ease, box-shadow 0.2s ease;
	}

	.submission-card.expanded {
		box-shadow: var(--shadow-sm);
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

	.meta-sep {
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
		gap: 10px;
		flex-wrap: wrap;
	}

	.submission-date {
		font-size: 12px;
		color: var(--text-muted);
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.file-badge {
		font-size: 12px;
		color: var(--text-secondary);
		font-family: 'JetBrains Mono', monospace;
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.file-size {
		color: var(--text-muted);
	}

	.status-badge {
		padding: 2px 8px;
		border-radius: 12px;
		font-size: 11px;
		text-transform: capitalize;
		font-weight: 500;
	}

	.review-badge {
		padding: 2px 8px;
		background: rgba(163, 113, 247, 0.12);
		color: var(--purple);
		border-radius: 12px;
		font-size: 11px;
		display: flex;
		align-items: center;
		gap: 4px;
	}

	.toggle-icon {
		color: var(--text-muted);
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
		color: var(--purple);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		margin-bottom: 10px;
		display: flex;
		align-items: center;
		gap: 6px;
	}

	.review-section {
		margin-top: 16px;
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

	.no-review {
		font-size: 13px;
		color: var(--text-muted);
		font-style: italic;
		margin-top: 12px;
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

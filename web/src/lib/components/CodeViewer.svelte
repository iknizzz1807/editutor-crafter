<script lang="ts">
	let {
		content,
		language,
		fileName,
		size,
		binary = false,
		truncated = false
	}: {
		content: string | null;
		language: string;
		fileName: string;
		size: number;
		binary?: boolean;
		truncated?: boolean;
	} = $props();

	function formatSize(bytes: number): string {
		if (bytes < 1024) return bytes + ' B';
		if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
		return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
	}

	let lines = $derived(content ? content.split('\n') : []);
</script>

<div class="code-viewer">
	<div class="file-header">
		<span class="file-name">{fileName}</span>
		<span class="file-meta">{formatSize(size)}</span>
	</div>

	{#if binary}
		<div class="notice">
			<span class="notice-icon">⚠️</span>
			Binary file — cannot display content
		</div>
	{:else if truncated}
		<div class="notice">
			<span class="notice-icon">⚠️</span>
			File too large to display ({formatSize(size)}). Max 100KB.
		</div>
	{:else if content}
		<div class="code-container">
			<div class="line-numbers">
				{#each lines as _, i}
					<span class="line-num">{i + 1}</span>
				{/each}
			</div>
			<pre class="code-content"><code>{content}</code></pre>
		</div>
	{:else}
		<div class="notice empty">Select a file from the tree to view its contents</div>
	{/if}
</div>

<style>
	.code-viewer {
		display: flex;
		flex-direction: column;
		height: 100%;
		overflow: hidden;
	}

	.file-header {
		display: flex;
		align-items: center;
		justify-content: space-between;
		padding: 6px 12px;
		background: var(--bg-card);
		border-bottom: 1px solid var(--border);
		flex-shrink: 0;
	}

	.file-name {
		font-size: 12px;
		font-family: 'JetBrains Mono', monospace;
		color: var(--text-primary);
		font-weight: 500;
		overflow: hidden;
		text-overflow: ellipsis;
		white-space: nowrap;
	}

	.file-meta {
		font-size: 11px;
		color: var(--text-muted);
		flex-shrink: 0;
		margin-left: 8px;
	}

	.notice {
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 8px;
		padding: 32px 16px;
		color: var(--text-muted);
		font-size: 13px;
		flex: 1;
	}

	.notice.empty {
		font-style: italic;
	}

	.code-container {
		display: flex;
		flex: 1;
		overflow: auto;
	}

	.line-numbers {
		display: flex;
		flex-direction: column;
		padding: 8px 0;
		background: var(--bg-sidebar);
		border-right: 1px solid var(--border);
		flex-shrink: 0;
		user-select: none;
	}

	.line-num {
		display: block;
		padding: 0 10px;
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		line-height: 1.5;
		color: var(--text-muted);
		text-align: right;
		min-width: 40px;
	}

	.code-content {
		flex: 1;
		margin: 0;
		padding: 8px 12px;
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		line-height: 1.5;
		color: var(--text-primary);
		tab-size: 4;
		overflow-x: auto;
		background: transparent;
	}

	.code-content code {
		display: block;
		white-space: pre;
	}
</style>

<script lang="ts">
	import { highlightCode } from '$lib/highlight.js';

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

	let highlightedHtml = $state<string | null>(null);

	$effect(() => {
		if (content && language) {
			highlightedHtml = null;
			highlightCode(content, language)
				.then((html) => {
					highlightedHtml = html;
				})
				.catch(() => {
					highlightedHtml = null;
				});
		} else {
			highlightedHtml = null;
		}
	});
</script>

<div class="code-viewer">
	<div class="file-header">
		<span class="file-name">{fileName}</span>
		<div class="file-header-right">
			{#if language && language !== 'text'}
				<span class="lang-badge">{language}</span>
			{/if}
			<span class="file-meta">{formatSize(size)}</span>
		</div>
	</div>

	{#if binary}
		<div class="notice">
			<span class="notice-icon">
				<svg width="16" height="16" viewBox="0 0 16 16" fill="var(--orange)">
					<path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"/>
				</svg>
			</span>
			Binary file — cannot display content
		</div>
	{:else if truncated}
		<div class="notice">
			<span class="notice-icon">
				<svg width="16" height="16" viewBox="0 0 16 16" fill="var(--orange)">
					<path fill-rule="evenodd" d="M8.22 1.754a.25.25 0 00-.44 0L1.698 13.132a.25.25 0 00.22.368h12.164a.25.25 0 00.22-.368L8.22 1.754zm-1.763-.707c.659-1.234 2.427-1.234 3.086 0l6.082 11.378A1.75 1.75 0 0114.082 15H1.918a1.75 1.75 0 01-1.543-2.575L6.457 1.047zM9 11a1 1 0 11-2 0 1 1 0 012 0zm-.25-5.25a.75.75 0 00-1.5 0v2.5a.75.75 0 001.5 0v-2.5z"/>
				</svg>
			</span>
			File too large to display ({formatSize(size)}). Max 100KB.
		</div>
	{:else if content}
		<div class="code-container">
			<div class="line-numbers">
				{#each lines as _, i}
					<span class="line-num">{i + 1}</span>
				{/each}
			</div>
			{#if highlightedHtml}
				<pre class="code-content highlighted"><code>{@html highlightedHtml}</code></pre>
			{:else}
				<pre class="code-content"><code>{content}</code></pre>
			{/if}
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

	.file-header-right {
		display: flex;
		align-items: center;
		gap: 8px;
		flex-shrink: 0;
	}

	.lang-badge {
		font-size: 10px;
		padding: 1px 6px;
		border-radius: 8px;
		background: rgba(88, 166, 255, 0.12);
		color: var(--blue);
		text-transform: lowercase;
		font-family: 'JetBrains Mono', monospace;
	}

	.file-meta {
		font-size: 11px;
		color: var(--text-muted);
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

	/* Shiki highlighted lines use spans with style attributes */
	.code-content.highlighted code :global(.line) {
		display: block;
	}
</style>

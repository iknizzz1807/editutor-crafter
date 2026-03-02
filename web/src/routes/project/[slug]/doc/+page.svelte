<script lang="ts">
	let { data } = $props();

	function scrollToSection(id: string) {
		const el = document.getElementById(id);
		if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
	}

	function handleContentClick(e: MouseEvent) {
		const target = e.target as HTMLElement;
		// Click on SVG image → open in new tab
		if (target.tagName === 'IMG') {
			const img = target as HTMLImageElement;
			if (img.src.includes('.svg') || img.src.includes('architecture-doc/asset')) {
				e.preventDefault();
				window.open(img.src, '_blank');
			}
		}
	}
</script>

<svelte:head>
	<title>{data.title} — Architecture Doc</title>
</svelte:head>

<div class="doc-page">
	<nav class="doc-toc">
		<div class="toc-header">
			<a href="javascript:history.back()" class="back-btn">
				<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor">
					<path fill-rule="evenodd" d="M7.78 12.53a.75.75 0 01-1.06 0L2.47 8.28a.75.75 0 010-1.06l4.25-4.25a.75.75 0 011.06 1.06L4.81 7h7.44a.75.75 0 010 1.5H4.81l2.97 2.97a.75.75 0 010 1.06z"/>
				</svg>
				Back
			</a>
			<div class="toc-title">Contents</div>
		</div>
		<div class="toc-entries">
			{#each data.toc as entry}
				<button
					class="toc-item level-{entry.level}"
					onclick={() => scrollToSection(entry.id)}
				>
					{entry.text}
				</button>
			{/each}
		</div>
	</nav>

	<main class="doc-content" onclick={handleContentClick} role="presentation">
		{@html data.html}
	</main>
</div>

<style>
	:global(body) {
		margin: 0;
		background: #0d1117;
		color: #c9d1d9;
		font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
	}

	.doc-page {
		display: flex;
		height: 100vh;
		overflow: hidden;
	}

	.doc-toc {
		width: 240px;
		flex-shrink: 0;
		background: #161b22;
		border-right: 1px solid #30363d;
		display: flex;
		flex-direction: column;
		overflow: hidden;
	}

	.toc-header {
		padding: 16px;
		border-bottom: 1px solid #30363d;
		display: flex;
		flex-direction: column;
		gap: 12px;
	}

	.back-btn {
		display: flex;
		align-items: center;
		gap: 6px;
		color: #8b949e;
		font-size: 12px;
		text-decoration: none;
		transition: color 0.15s;
	}

	.back-btn:hover {
		color: #c9d1d9;
	}

	.toc-title {
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		color: #8b949e;
	}

	.toc-entries {
		flex: 1;
		overflow-y: auto;
		padding: 8px;
	}

	.toc-item {
		display: block;
		width: 100%;
		text-align: left;
		padding: 4px 8px;
		font-size: 12px;
		color: #8b949e;
		background: none;
		border: none;
		border-radius: 4px;
		cursor: pointer;
		font-family: inherit;
		line-height: 1.4;
		transition: all 0.1s;
	}

	.toc-item:hover {
		background: #21262d;
		color: #c9d1d9;
	}

	.toc-item.level-2 { padding-left: 16px; }
	.toc-item.level-3 { padding-left: 24px; font-size: 11px; }
	.toc-item.level-4 { padding-left: 32px; font-size: 11px; }

	.doc-content {
		flex: 1;
		min-width: 0;
		overflow-y: auto;
		padding: 32px 48px;
		font-size: 14px;
		line-height: 1.7;
		color: #c9d1d9;
	}

	/* SVG images: clickable cursor to hint they open in new tab */
	.doc-content :global(img[src*=".svg"]),
	.doc-content :global(img[src*="asset"]) {
		cursor: pointer;
		max-width: 100%;
		height: auto;
		border-radius: 6px;
		margin: 16px 0;
		display: block;
		border: 1px solid #30363d;
		transition: border-color 0.15s;
	}

	.doc-content :global(img[src*=".svg"]:hover),
	.doc-content :global(img[src*="asset"]:hover) {
		border-color: #58a6ff;
	}

	.doc-content :global(h1) {
		font-size: 24px;
		font-weight: 700;
		color: #f0f6fc;
		margin: 0 0 20px 0;
		padding-bottom: 10px;
		border-bottom: 1px solid #30363d;
	}

	.doc-content :global(h2) {
		font-size: 20px;
		font-weight: 600;
		color: #f0f6fc;
		margin: 36px 0 14px 0;
		padding-bottom: 6px;
		border-bottom: 1px solid #30363d;
	}

	.doc-content :global(h3) {
		font-size: 16px;
		font-weight: 600;
		color: #f0f6fc;
		margin: 24px 0 10px 0;
	}

	.doc-content :global(h4) {
		font-size: 14px;
		font-weight: 600;
		color: #f0f6fc;
		margin: 20px 0 8px 0;
	}

	.doc-content :global(p) { margin: 0 0 14px 0; }

	.doc-content :global(ul),
	.doc-content :global(ol) {
		margin: 0 0 14px 0;
		padding-left: 24px;
	}

	.doc-content :global(li) { margin-bottom: 4px; }

	.doc-content :global(code) {
		font-family: 'JetBrains Mono', monospace;
		background: #161b22;
		border: 1px solid #30363d;
		padding: 1px 6px;
		border-radius: 4px;
		font-size: 12px;
		color: #58a6ff;
	}

	.doc-content :global(.code-block-wrapper) {
		position: relative;
		margin: 14px 0;
	}

	.doc-content :global(.code-lang) {
		position: absolute;
		top: 6px;
		right: 8px;
		font-size: 10px;
		color: #8b949e;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-family: 'Inter', sans-serif;
		z-index: 1;
	}

	.doc-content :global(.arch-pre) {
		background: #161b22;
		border: 1px solid #30363d;
		padding: 14px;
		border-radius: 6px;
		overflow-x: auto;
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		white-space: pre-wrap;
		line-height: 1.5;
		color: #c9d1d9;
		margin: 0;
	}

	.doc-content :global(.arch-pre code) {
		background: none;
		padding: 0;
		border: none;
		color: inherit;
		font-family: inherit;
	}

	.doc-content :global(strong) {
		color: #f0f6fc;
		font-weight: 600;
	}

	.doc-content :global(blockquote) {
		margin: 14px 0;
		padding: 8px 16px;
		border-left: 3px solid #58a6ff;
		background: rgba(88, 166, 255, 0.05);
		border-radius: 0 6px 6px 0;
		color: #8b949e;
	}

	.doc-content :global(table) {
		width: 100%;
		border-collapse: collapse;
		margin: 14px 0;
		font-size: 13px;
	}

	.doc-content :global(th) {
		text-align: left;
		padding: 8px 12px;
		background: #161b22;
		border: 1px solid #30363d;
		font-weight: 600;
		color: #f0f6fc;
	}

	.doc-content :global(td) {
		padding: 8px 12px;
		border: 1px solid #30363d;
	}

	.doc-content :global(hr) {
		border: none;
		border-top: 1px solid #30363d;
		margin: 24px 0;
	}

	.doc-content :global(a) {
		color: #58a6ff;
		text-decoration: none;
	}

	.doc-content :global(a:hover) {
		text-decoration: underline;
	}
</style>

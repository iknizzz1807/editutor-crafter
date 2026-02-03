<script lang="ts">
	let {
		projectSlug,
		hasDoc
	}: {
		projectSlug: string;
		hasDoc: boolean;
	} = $props();

	let isOpen = $state(false);
	let loading = $state(false);
	let errorMsg = $state('');
	let html = $state('');
	let markdown = $state('');
	let title = $state('');
	let toc = $state<Array<{ level: number; text: string; id: string }>>([]);
	let fetched = false;

	async function toggle() {
		if (!hasDoc) return;
		isOpen = !isOpen;
		if (isOpen && !fetched) {
			loading = true;
			errorMsg = '';
			try {
				const res = await fetch(`/api/project/${projectSlug}/architecture-doc`);
				if (!res.ok) throw new Error('Failed to load document');
				const data = await res.json();
				html = data.html;
				markdown = data.markdown;
				title = data.title;
				toc = data.toc || [];
				fetched = true;
			} catch (e) {
				errorMsg = e instanceof Error ? e.message : 'Failed to load';
			} finally {
				loading = false;
			}
		}
	}

	function downloadMd() {
		const blob = new Blob([markdown], { type: 'text/markdown' });
		const url = URL.createObjectURL(blob);
		const a = document.createElement('a');
		a.href = url;
		a.download = `${projectSlug}-architecture.md`;
		a.click();
		URL.revokeObjectURL(url);
	}

	function scrollToSection(id: string) {
		const el = document.getElementById(id);
		if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
	}
</script>

{#if hasDoc}
	<div class="arch-doc" class:open={isOpen}>
		<button class="arch-toggle" onclick={toggle}>
			<div class="arch-toggle-left">
				<div class="arch-icon">
					<svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor">
						<path d="M0 1.75A.75.75 0 01.75 1h4.253c1.227 0 2.317.59 3 1.501A3.744 3.744 0 0111.006 1h4.245a.75.75 0 01.75.75v10.5a.75.75 0 01-.75.75h-4.507a2.25 2.25 0 00-1.591.659l-.622.621a.75.75 0 01-1.06 0l-.622-.621A2.25 2.25 0 005.258 13H.75a.75.75 0 01-.75-.75V1.75zm7.251 10.324l.004-5.073-.002-2.253A2.25 2.25 0 005.003 2.5H1.5v9h3.757a3.75 3.75 0 011.994.574zM8.755 4.75l-.004 7.322a3.752 3.752 0 011.992-.572H14.5v-9h-3.495a2.25 2.25 0 00-2.25 2.25z"/>
					</svg>
				</div>
				<span class="arch-label">Architecture Document</span>
			</div>
			<span class="arch-chevron" class:open={isOpen}>
				<svg width="12" height="12" viewBox="0 0 12 12" fill="currentColor">
					<path d="M2.5 4.5L6 8L9.5 4.5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round" fill="none"/>
				</svg>
			</span>
		</button>

		{#if isOpen}
			<div class="arch-body">
				{#if loading}
					<div class="arch-loading">
						<div class="spinner"></div>
						<span>Loading architecture document...</span>
					</div>
				{:else if errorMsg}
					<div class="arch-error">{errorMsg}</div>
				{:else}
					<div class="arch-toolbar">
						<span class="arch-title">{title}</span>
						<div class="arch-actions">
							<button class="arch-btn" onclick={downloadMd}>
								<svg width="14" height="14" viewBox="0 0 16 16" fill="currentColor"><path d="M2.75 14A1.75 1.75 0 0 1 1 12.25v-2.5a.75.75 0 0 1 1.5 0v2.5c0 .138.112.25.25.25h10.5a.25.25 0 0 0 .25-.25v-2.5a.75.75 0 0 1 1.5 0v2.5A1.75 1.75 0 0 1 13.25 14ZM7.25 7.689V2a.75.75 0 0 1 1.5 0v5.689l1.97-1.969a.749.749 0 1 1 1.06 1.06l-3.25 3.25a.749.749 0 0 1-1.06 0L4.22 6.78a.749.749 0 1 1 1.06-1.06l1.97 1.969Z"/></svg>
								Download .md
							</button>
						</div>
					</div>

					<div class="arch-layout">
						{#if toc.length > 0}
							<nav class="arch-toc">
								<div class="toc-title">Contents</div>
								{#each toc as entry}
									<button
										class="toc-item level-{entry.level}"
										onclick={() => scrollToSection(entry.id)}
									>
										{entry.text}
									</button>
								{/each}
							</nav>
						{/if}
						<div class="arch-content">
							{@html html}
						</div>
					</div>
				{/if}
			</div>
		{/if}
	</div>
{/if}

<style>
	.arch-doc {
		margin-bottom: 20px;
		border: 1px solid var(--border);
		border-radius: var(--radius-md);
		overflow: hidden;
		background: var(--bg-card);
	}

	.arch-doc.open {
		border-color: rgba(163, 113, 247, 0.4);
	}

	.arch-toggle {
		width: 100%;
		padding: 14px 18px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		background: var(--bg-sidebar);
		border: none;
		cursor: pointer;
		color: var(--text-primary);
		font-family: inherit;
		font-size: 14px;
		font-weight: 600;
		transition: background 0.15s;
	}

	.arch-toggle:hover {
		background: var(--bg-card);
	}

	.arch-toggle-left {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.arch-icon {
		width: 28px;
		height: 28px;
		border-radius: 6px;
		display: flex;
		align-items: center;
		justify-content: center;
		background: rgba(163, 113, 247, 0.12);
		color: var(--purple);
	}

	.arch-label {
		color: var(--text-primary);
	}

	.arch-chevron {
		color: var(--text-muted);
		transition: transform 0.2s ease;
		display: flex;
	}

	.arch-chevron.open {
		transform: rotate(180deg);
	}

	.arch-body {
		border-top: 1px solid var(--border);
	}

	.arch-loading {
		padding: 40px;
		display: flex;
		align-items: center;
		justify-content: center;
		gap: 12px;
		color: var(--text-muted);
		font-size: 13px;
	}

	.spinner {
		width: 18px;
		height: 18px;
		border: 2px solid var(--border);
		border-top-color: var(--purple);
		border-radius: 50%;
		animation: spin 0.6s linear infinite;
	}

	@keyframes spin {
		to { transform: rotate(360deg); }
	}

	.arch-error {
		padding: 20px;
		color: var(--red);
		font-size: 13px;
		text-align: center;
	}

	.arch-toolbar {
		padding: 12px 18px;
		display: flex;
		align-items: center;
		justify-content: space-between;
		border-bottom: 1px solid var(--border);
		background: var(--bg-sidebar);
	}

	.arch-title {
		font-size: 14px;
		font-weight: 600;
		color: var(--text-primary);
	}

	.arch-actions {
		display: flex;
		gap: 8px;
	}

	.arch-btn {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 5px 12px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-radius: 6px;
		color: var(--text-secondary);
		font-size: 12px;
		font-family: inherit;
		cursor: pointer;
		transition: all 0.15s;
	}

	.arch-btn:hover {
		background: var(--bg-elevated);
		color: var(--text-primary);
		border-color: var(--text-muted);
	}

	.arch-layout {
		display: flex;
		min-height: 300px;
	}

	.arch-toc {
		width: 220px;
		flex-shrink: 0;
		padding: 16px;
		border-right: 1px solid var(--border);
		background: var(--bg-sidebar);
		overflow-y: auto;
		max-height: 70vh;
		position: sticky;
		top: 0;
	}

	.toc-title {
		font-size: 11px;
		font-weight: 600;
		text-transform: uppercase;
		letter-spacing: 0.5px;
		color: var(--text-muted);
		margin-bottom: 10px;
	}

	.toc-item {
		display: block;
		width: 100%;
		text-align: left;
		padding: 4px 8px;
		font-size: 12px;
		color: var(--text-secondary);
		background: none;
		border: none;
		border-radius: 4px;
		cursor: pointer;
		font-family: inherit;
		line-height: 1.4;
		transition: all 0.1s;
	}

	.toc-item:hover {
		background: var(--bg-card);
		color: var(--text-primary);
	}

	.toc-item.level-2 { padding-left: 16px; }
	.toc-item.level-3 { padding-left: 24px; font-size: 11px; }
	.toc-item.level-4 { padding-left: 32px; font-size: 11px; }

	.arch-content {
		flex: 1;
		padding: 24px 28px;
		overflow-x: auto;
		max-height: 70vh;
		overflow-y: auto;
		font-size: 14px;
		line-height: 1.7;
		color: var(--text-secondary);
	}

	/* Markdown content styles */
	.arch-content :global(h1) {
		font-size: 22px;
		font-weight: 700;
		color: var(--text-primary);
		margin: 0 0 16px 0;
		padding-bottom: 8px;
		border-bottom: 1px solid var(--border);
	}

	.arch-content :global(h2) {
		font-size: 18px;
		font-weight: 600;
		color: var(--text-primary);
		margin: 28px 0 12px 0;
		padding-bottom: 6px;
		border-bottom: 1px solid var(--border);
	}

	.arch-content :global(h3) {
		font-size: 15px;
		font-weight: 600;
		color: var(--text-primary);
		margin: 22px 0 8px 0;
	}

	.arch-content :global(h4) {
		font-size: 14px;
		font-weight: 600;
		color: var(--text-primary);
		margin: 18px 0 6px 0;
	}

	.arch-content :global(p) {
		margin: 0 0 12px 0;
	}

	.arch-content :global(ul),
	.arch-content :global(ol) {
		margin: 0 0 12px 0;
		padding-left: 24px;
	}

	.arch-content :global(li) {
		margin-bottom: 4px;
	}

	.arch-content :global(code) {
		font-family: 'JetBrains Mono', monospace;
		background: var(--bg-dark);
		border: 1px solid var(--border-light);
		padding: 1px 6px;
		border-radius: 4px;
		font-size: 12px;
		color: var(--blue);
	}

	.arch-content :global(.code-block-wrapper) {
		position: relative;
		margin: 12px 0;
	}

	.arch-content :global(.code-lang) {
		position: absolute;
		top: 6px;
		right: 8px;
		font-size: 10px;
		color: var(--text-muted);
		text-transform: uppercase;
		letter-spacing: 0.5px;
		font-family: 'Inter', sans-serif;
		z-index: 1;
	}

	.arch-content :global(.arch-pre) {
		background: var(--bg-dark);
		border: 1px solid var(--border);
		padding: 14px;
		border-radius: var(--radius-sm);
		overflow-x: auto;
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		white-space: pre-wrap;
		line-height: 1.5;
		color: var(--text-primary);
		margin: 0;
	}

	.arch-content :global(.arch-pre code) {
		background: none;
		padding: 0;
		border: none;
		color: inherit;
		font-family: inherit;
	}

	.arch-content :global(.shiki-highlighted .line) {
		display: block;
	}

	.arch-content :global(strong) {
		color: var(--text-primary);
		font-weight: 600;
	}

	.arch-content :global(em) {
		font-style: italic;
	}

	.arch-content :global(blockquote) {
		margin: 12px 0;
		padding: 8px 16px;
		border-left: 3px solid var(--purple);
		background: rgba(163, 113, 247, 0.05);
		border-radius: 0 6px 6px 0;
		color: var(--text-muted);
	}

	.arch-content :global(table) {
		width: 100%;
		border-collapse: collapse;
		margin: 12px 0;
		font-size: 13px;
	}

	.arch-content :global(th) {
		text-align: left;
		padding: 8px 12px;
		background: var(--bg-sidebar);
		border: 1px solid var(--border);
		font-weight: 600;
		color: var(--text-primary);
	}

	.arch-content :global(td) {
		padding: 8px 12px;
		border: 1px solid var(--border);
	}

	.arch-content :global(hr) {
		border: none;
		border-top: 1px solid var(--border);
		margin: 20px 0;
	}

	.arch-content :global(img) {
		max-width: 100%;
		border-radius: var(--radius-sm);
		margin: 12px 0;
	}

	.arch-content :global(a) {
		color: var(--blue);
		text-decoration: none;
	}

	.arch-content :global(a:hover) {
		text-decoration: underline;
	}

	@media (max-width: 768px) {
		.arch-layout {
			flex-direction: column;
		}

		.arch-toc {
			width: 100%;
			max-height: 200px;
			border-right: none;
			border-bottom: 1px solid var(--border);
		}

		.arch-content {
			padding: 16px;
		}
	}
</style>

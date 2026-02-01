<script lang="ts">
	let {
		level1,
		level2,
		level3
	}: {
		level1: string | null;
		level2: string | null;
		level3: string | null;
	} = $props();

	let openLevel = $state<number | null>(null);

	function toggle(level: number) {
		openLevel = openLevel === level ? null : level;
	}

	const levels = $derived(
		[
			{ level: 1, label: 'Level 1 - Gentle Nudge', icon: '🌱', content: level1 },
			{ level: 2, label: 'Level 2 - More Detail', icon: '🌿', content: level2 },
			{ level: 3, label: 'Level 3 - Full Guidance', icon: '🌳', content: level3 }
		].filter((l) => l.content)
	);
</script>

<div class="hints-section">
	{#each levels as hint}
		<div class="hint-item" class:open={openLevel === hint.level}>
			<button class="hint-header" onclick={() => toggle(hint.level)}>
				<span class="level">
					<span class="level-icon">{hint.icon}</span>
					{hint.label}
				</span>
				<span class="toggle-icon">▼</span>
			</button>
			{#if openLevel === hint.level}
				<div class="hint-content">
					{@html formatHint(hint.content || '')}
				</div>
			{/if}
		</div>
	{/each}
</div>

<script lang="ts" module>
	function escapeHtml(text: string): string {
		return text
			.replace(/&/g, '&amp;')
			.replace(/</g, '&lt;')
			.replace(/>/g, '&gt;')
			.replace(/"/g, '&quot;');
	}

	function formatHint(text: string): string {
		if (!text) return '';

		const codeBlocks: string[] = [];
		text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_match, _lang, code) => {
			const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
			codeBlocks.push(`<pre><code>${escapeHtml(code)}</code></pre>`);
			return placeholder;
		});

		const inlineCode: string[] = [];
		text = text.replace(/`([^`]+)`/g, (_match, code) => {
			const placeholder = `__INLINE_CODE_${inlineCode.length}__`;
			inlineCode.push(`<code>${escapeHtml(code)}</code>`);
			return placeholder;
		});

		text = escapeHtml(text);

		codeBlocks.forEach((block, i) => {
			text = text.replace(`__CODE_BLOCK_${i}__`, block);
		});

		inlineCode.forEach((code, i) => {
			text = text.replace(`__INLINE_CODE_${i}__`, code);
		});

		text = text.replace(/\n/g, '<br>');
		return text;
	}
</script>

<style>
	.hints-section {
		background: var(--bg-sidebar);
		border-radius: 6px;
		overflow: hidden;
	}

	.hint-item {
		border-bottom: 1px solid var(--border);
	}

	.hint-item:last-child {
		border-bottom: none;
	}

	.hint-header {
		padding: 12px 14px;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: space-between;
		font-size: 13px;
		font-weight: 500;
		color: var(--text-secondary);
		transition: all 0.15s ease;
		width: 100%;
		background: none;
		border: none;
		font-family: inherit;
		text-align: left;
	}

	.hint-header:hover {
		background: var(--bg-card);
		color: var(--text-primary);
	}

	.level {
		display: flex;
		align-items: center;
		gap: 8px;
	}

	.level-icon {
		font-size: 14px;
	}

	.toggle-icon {
		transition: transform 0.2s ease;
	}

	.hint-item.open .toggle-icon {
		transform: rotate(180deg);
	}

	.hint-item.open .hint-header {
		background: var(--bg-card);
	}

	.hint-content {
		padding: 0 14px 14px;
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.7;
	}

	.hint-content :global(pre) {
		background: var(--bg-dark);
		padding: 12px;
		border-radius: 6px;
		overflow-x: auto;
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		margin-top: 8px;
		white-space: pre-wrap;
	}

	.hint-content :global(code) {
		font-family: 'JetBrains Mono', monospace;
		background: var(--bg-dark);
		padding: 2px 6px;
		border-radius: 4px;
		font-size: 12px;
	}

	.hint-content :global(pre code) {
		background: none;
		padding: 0;
	}
</style>

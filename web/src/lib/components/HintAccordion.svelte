<script lang="ts">
	import { highlightCode } from '$lib/highlight.js';

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
			{ level: 1, label: 'Gentle Nudge', content: level1, color: 'var(--accent-bright)', bg: 'rgba(63, 185, 80, 0.08)', border: 'rgba(63, 185, 80, 0.25)' },
			{ level: 2, label: 'More Detail', content: level2, color: 'var(--blue)', bg: 'rgba(88, 166, 255, 0.08)', border: 'rgba(88, 166, 255, 0.25)' },
			{ level: 3, label: 'Full Guidance', content: level3, color: 'var(--purple)', bg: 'rgba(163, 113, 247, 0.08)', border: 'rgba(163, 113, 247, 0.25)' }
		].filter((l) => l.content)
	);

	// Store highlighted HTML per level
	let highlightedContent = $state<Record<number, string>>({});

	$effect(() => {
		if (openLevel !== null) {
			const hint = levels.find((l) => l.level === openLevel);
			if (hint?.content && !highlightedContent[openLevel]) {
				formatHintAsync(hint.content, openLevel);
			}
		}
	});

	async function formatHintAsync(text: string, level: number) {
		const result = await formatHintWithHighlight(text);
		highlightedContent = { ...highlightedContent, [level]: result };
	}
</script>

<div class="hints-section">
	{#each levels as hint}
		{@const isOpen = openLevel === hint.level}
		<div
			class="hint-item"
			class:open={isOpen}
			style="--hint-color: {hint.color}; --hint-bg: {hint.bg}; --hint-border: {hint.border}"
		>
			<button class="hint-header" onclick={() => toggle(hint.level)}>
				<span class="level">
					<span class="level-num">{hint.level}</span>
					<span class="level-label">{hint.label}</span>
				</span>
				<span class="toggle-icon">
					<svg width="12" height="12" viewBox="0 0 16 16" fill="currentColor">
						<path fill-rule="evenodd" d="M12.78 6.22a.75.75 0 010 1.06l-4.25 4.25a.75.75 0 01-1.06 0L3.22 7.28a.75.75 0 011.06-1.06L8 9.94l3.72-3.72a.75.75 0 011.06 0z"/>
					</svg>
				</span>
			</button>
			{#if isOpen}
				<div class="hint-content">
					{#if highlightedContent[hint.level]}
						{@html highlightedContent[hint.level]}
					{:else}
						{@html formatHintBasic(hint.content || '')}
					{/if}
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

	// Detect if content is primarily code (no markdown fences)
	function looksLikeCode(text: string): boolean {
		const hasCodeFences = /```/.test(text);
		if (hasCodeFences) return false;

		const lines = text.split('\n').filter((l) => l.trim());
		if (lines.length < 3) return false;

		const codePattern =
			/[{};()=]|^\s*(function |const |let |var |class |import |export |def |fn |if \(|for \(|while \(|return |package |type |struct )/;
		const codeLines = lines.filter((l) => codePattern.test(l));
		return codeLines.length / lines.length > 0.3;
	}

	// Guess language from code content
	function guessLanguage(text: string): string {
		if (/\b(func |package |:=|fmt\.)/.test(text)) return 'go';
		if (/\b(def |import |print\(|self\.)/.test(text)) return 'python';
		if (/\b(fn |let mut |impl |pub )/.test(text)) return 'rust';
		if (/\b(public static|System\.out|void )/.test(text)) return 'java';
		if (/\b(interface |type .*= )/.test(text) && /:\s*(string|number|boolean)/.test(text))
			return 'typescript';
		return 'javascript';
	}

	function formatHintBasic(text: string): string {
		if (!text) return '';

		// Auto-detect pure code content
		if (looksLikeCode(text)) {
			return `<div class="code-block-wrapper"><pre class="hint-pre"><code>${escapeHtml(text)}</code></pre></div>`;
		}

		const codeBlocks: string[] = [];
		text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_match, lang, code) => {
			const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
			const langLabel = lang ? `<span class="code-lang">${escapeHtml(lang)}</span>` : '';
			codeBlocks.push(
				`<div class="code-block-wrapper">${langLabel}<pre class="hint-pre"><code>${escapeHtml(code.trim())}</code></pre></div>`
			);
			return placeholder;
		});

		const inlineCode: string[] = [];
		text = text.replace(/`([^`]+)`/g, (_match, code) => {
			const placeholder = `__INLINE_CODE_${inlineCode.length}__`;
			inlineCode.push(`<code class="hint-inline-code">${escapeHtml(code)}</code>`);
			return placeholder;
		});

		text = escapeHtml(text);

		codeBlocks.forEach((block, i) => {
			text = text.replace(`__CODE_BLOCK_${i}__`, block);
		});

		inlineCode.forEach((code, i) => {
			text = text.replace(`__INLINE_CODE_${i}__`, code);
		});

		text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
		text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
		text = text.replace(
			/^([-*])\s+(.+)$/gm,
			'<div class="hint-bullet"><span class="hint-bullet-dot"></span><span>$2</span></div>'
		);
		text = text.replace(
			/^(\d+)\.\s+(.+)$/gm,
			'<div class="hint-bullet"><span class="hint-bullet-num">$1.</span><span>$2</span></div>'
		);
		text = text.replace(/\n/g, '<br>');
		return text;
	}

	async function formatHintWithHighlight(text: string): Promise<string> {
		if (!text) return '';

		const { highlightCode: hl } = await import('$lib/highlight.js');

		// Auto-detect pure code content
		if (looksLikeCode(text)) {
			const lang = guessLanguage(text);
			const highlighted = await hl(text, lang);
			const langLabel = `<span class="code-lang">${escapeHtml(lang)}</span>`;
			return `<div class="code-block-wrapper">${langLabel}<pre class="hint-pre shiki-highlighted"><code>${highlighted}</code></pre></div>`;
		}

		const codeBlocks: { lang: string; code: string }[] = [];
		text = text.replace(/```(\w*)\n?([\s\S]*?)```/g, (_match, lang, code) => {
			const placeholder = `__CODE_BLOCK_${codeBlocks.length}__`;
			codeBlocks.push({ lang: lang || '', code: code.trim() });
			return placeholder;
		});

		const inlineCode: string[] = [];
		text = text.replace(/`([^`]+)`/g, (_match, code) => {
			const placeholder = `__INLINE_CODE_${inlineCode.length}__`;
			inlineCode.push(`<code class="hint-inline-code">${escapeHtml(code)}</code>`);
			return placeholder;
		});

		text = escapeHtml(text);

		for (let i = 0; i < codeBlocks.length; i++) {
			const { lang, code } = codeBlocks[i];
			const highlighted = await hl(code, lang || undefined);
			const langLabel = lang ? `<span class="code-lang">${escapeHtml(lang)}</span>` : '';
			text = text.replace(
				`__CODE_BLOCK_${i}__`,
				`<div class="code-block-wrapper">${langLabel}<pre class="hint-pre shiki-highlighted"><code>${highlighted}</code></pre></div>`
			);
		}

		inlineCode.forEach((code, i) => {
			text = text.replace(`__INLINE_CODE_${i}__`, code);
		});

		text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
		text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
		text = text.replace(
			/^([-*])\s+(.+)$/gm,
			'<div class="hint-bullet"><span class="hint-bullet-dot"></span><span>$2</span></div>'
		);
		text = text.replace(
			/^(\d+)\.\s+(.+)$/gm,
			'<div class="hint-bullet"><span class="hint-bullet-num">$1.</span><span>$2</span></div>'
		);
		text = text.replace(/\n/g, '<br>');
		return text;
	}
</script>

<style>
	.hints-section {
		border-radius: var(--radius-md);
		overflow: hidden;
		border: 1px solid var(--border);
	}

	.hint-item {
		border-bottom: 1px solid var(--border);
	}

	.hint-item:last-child {
		border-bottom: none;
	}

	.hint-header {
		padding: 10px 14px;
		cursor: pointer;
		display: flex;
		align-items: center;
		justify-content: space-between;
		font-size: 13px;
		font-weight: 500;
		color: var(--text-secondary);
		transition: all 0.15s ease;
		width: 100%;
		background: var(--bg-sidebar);
		border: none;
		font-family: inherit;
		text-align: left;
	}

	.hint-header:hover {
		background: var(--bg-card);
		color: var(--text-primary);
	}

	.hint-item.open .hint-header {
		background: var(--hint-bg);
		color: var(--hint-color);
		border-bottom: 1px solid var(--hint-border);
	}

	.level {
		display: flex;
		align-items: center;
		gap: 10px;
	}

	.level-num {
		width: 22px;
		height: 22px;
		border-radius: 50%;
		display: flex;
		align-items: center;
		justify-content: center;
		font-size: 11px;
		font-weight: 700;
		background: var(--hint-bg);
		color: var(--hint-color);
		border: 1px solid var(--hint-border);
		flex-shrink: 0;
	}

	.hint-item.open .level-num {
		background: var(--hint-color);
		color: white;
		border-color: var(--hint-color);
	}

	.level-label {
		font-weight: 500;
	}

	.toggle-icon {
		transition: transform 0.2s ease;
		color: var(--text-muted);
		display: flex;
	}

	.hint-item.open .toggle-icon {
		transform: rotate(180deg);
		color: var(--hint-color);
	}

	.hint-content {
		padding: 14px 14px 14px 46px;
		font-size: 13px;
		color: var(--text-secondary);
		line-height: 1.7;
		background: var(--bg-card);
		border-left: 3px solid var(--hint-color);
	}

	.hint-content :global(strong) {
		color: var(--text-primary);
		font-weight: 600;
	}

	.hint-content :global(em) {
		color: var(--text-secondary);
		font-style: italic;
	}

	.hint-content :global(.hint-inline-code) {
		font-family: 'JetBrains Mono', monospace;
		background: var(--bg-dark);
		border: 1px solid var(--border-light);
		padding: 1px 6px;
		border-radius: 4px;
		font-size: 12px;
		color: var(--blue);
	}

	.hint-content :global(.code-block-wrapper) {
		position: relative;
		margin: 10px 0;
	}

	.hint-content :global(.code-lang) {
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

	.hint-content :global(.hint-pre) {
		background: var(--bg-dark);
		border: 1px solid var(--border);
		padding: 12px;
		border-radius: var(--radius-sm);
		overflow-x: auto;
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		white-space: pre-wrap;
		line-height: 1.5;
		color: var(--text-primary);
		margin: 0;
	}

	.hint-content :global(.hint-pre code) {
		background: none;
		padding: 0;
		border: none;
		color: inherit;
		font-family: inherit;
	}

	/* Shiki highlighted spans */
	.hint-content :global(.shiki-highlighted .line) {
		display: block;
	}

	.hint-content :global(.hint-bullet) {
		display: flex;
		align-items: flex-start;
		gap: 8px;
		padding: 3px 0;
	}

	.hint-content :global(.hint-bullet-dot) {
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: var(--hint-color);
		flex-shrink: 0;
		margin-top: 8px;
	}

	.hint-content :global(.hint-bullet-num) {
		color: var(--hint-color);
		font-weight: 600;
		flex-shrink: 0;
		min-width: 18px;
	}
</style>

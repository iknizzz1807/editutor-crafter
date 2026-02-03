<script lang="ts">
	import { highlightCode } from '$lib/highlight.js';

	let { review }: { review: string } = $props();

	interface ParsedBlock {
		type: 'heading' | 'bullet' | 'numbered' | 'code' | 'text';
		content: string;
		level?: number;
		lang?: string;
	}

	function parseReview(text: string): ParsedBlock[] {
		const lines = text.split('\n');
		const blocks: ParsedBlock[] = [];
		let inCode = false;
		let codeBuffer: string[] = [];
		let codeLang = '';

		for (const line of lines) {
			if (line.startsWith('```')) {
				if (inCode) {
					blocks.push({ type: 'code', content: codeBuffer.join('\n'), lang: codeLang });
					codeBuffer = [];
					codeLang = '';
					inCode = false;
				} else {
					codeLang = line.slice(3).trim();
					inCode = true;
				}
				continue;
			}

			if (inCode) {
				codeBuffer.push(line);
				continue;
			}

			const headingMatch = line.match(/^(#{1,3})\s+(.+)/);
			if (headingMatch) {
				blocks.push({
					type: 'heading',
					content: headingMatch[2],
					level: headingMatch[1].length
				});
				continue;
			}

			if (line.match(/^[-*]\s+/)) {
				blocks.push({ type: 'bullet', content: line.replace(/^[-*]\s+/, '') });
				continue;
			}

			const numMatch = line.match(/^\d+\.\s+/);
			if (numMatch) {
				blocks.push({ type: 'numbered', content: line.replace(/^\d+\.\s+/, '') });
				continue;
			}

			if (line.trim()) {
				blocks.push({ type: 'text', content: line });
			}
		}

		if (inCode && codeBuffer.length > 0) {
			blocks.push({ type: 'code', content: codeBuffer.join('\n'), lang: codeLang });
		}

		return blocks;
	}

	function formatInline(text: string): string {
		return text
			.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
			.replace(/`(.+?)`/g, '<code class="inline-code">$1</code>')
			.replace(/\*(.+?)\*/g, '<em>$1</em>');
	}

	let blocks = $derived(parseReview(review));

	// Async highlighted code blocks
	let highlightedBlocks = $state<Record<number, string>>({});

	$effect(() => {
		const codeBlocks = blocks
			.map((b, i) => ({ block: b, index: i }))
			.filter((b) => b.block.type === 'code');

		for (const { block, index } of codeBlocks) {
			if (!highlightedBlocks[index]) {
				highlightCode(block.content, block.lang || undefined).then((html) => {
					highlightedBlocks = { ...highlightedBlocks, [index]: html };
				});
			}
		}
	});
</script>

<div class="ai-review">
	<div class="review-header">
		<svg width="16" height="16" viewBox="0 0 16 16" fill="var(--purple)">
			<path fill-rule="evenodd" d="M8 0a8 8 0 110 16A8 8 0 018 0zm-.5 4.75a.75.75 0 011.5 0v3.5a.75.75 0 01-.37.65l-2.5 1.5a.75.75 0 01-.76-1.3L7.5 7.96V4.75z"/>
		</svg>
		<span class="review-title">AI Review</span>
	</div>
	<div class="review-body">
		{#each blocks as block, idx}
			{#if block.type === 'heading'}
				{#if block.level === 1}
					<h3 class="review-h1">{block.content}</h3>
				{:else if block.level === 2}
					<h4 class="review-h2">{block.content}</h4>
				{:else}
					<h5 class="review-h3">{block.content}</h5>
				{/if}
			{:else if block.type === 'bullet'}
				<div class="review-bullet">
					<span class="bullet-dot"></span>
					<span>{@html formatInline(block.content)}</span>
				</div>
			{:else if block.type === 'numbered'}
				<div class="review-bullet numbered">
					<span class="bullet-num"></span>
					<span>{@html formatInline(block.content)}</span>
				</div>
			{:else if block.type === 'code'}
				<div class="review-code-wrapper">
					{#if block.lang}
						<span class="review-code-lang">{block.lang}</span>
					{/if}
					{#if highlightedBlocks[idx]}
						<pre class="review-code shiki-highlighted"><code>{@html highlightedBlocks[idx]}</code></pre>
					{:else}
						<pre class="review-code"><code>{block.content}</code></pre>
					{/if}
				</div>
			{:else}
				<p class="review-text">{@html formatInline(block.content)}</p>
			{/if}
		{/each}
	</div>
</div>

<style>
	.ai-review {
		margin-top: 10px;
		background: var(--bg-card);
		border: 1px solid var(--border);
		border-left: 3px solid var(--purple);
		border-radius: 0 var(--radius-md) var(--radius-md) 0;
		overflow: hidden;
	}

	.review-header {
		display: flex;
		align-items: center;
		gap: 8px;
		padding: 10px 14px;
		background: rgba(163, 113, 247, 0.06);
		border-bottom: 1px solid var(--border);
	}

	.review-title {
		font-size: 12px;
		font-weight: 600;
		color: var(--purple);
		text-transform: uppercase;
		letter-spacing: 0.5px;
	}

	.review-body {
		padding: 14px;
		font-size: 13px;
		line-height: 1.7;
		color: var(--text-secondary);
	}

	.review-h1 {
		font-size: 15px;
		font-weight: 700;
		color: var(--text-primary);
		margin: 16px 0 8px;
		padding-bottom: 6px;
		border-bottom: 1px solid var(--border-light);
	}

	.review-h1:first-child {
		margin-top: 0;
	}

	.review-h2 {
		font-size: 14px;
		font-weight: 600;
		color: var(--text-primary);
		margin: 12px 0 6px;
	}

	.review-h3 {
		font-size: 13px;
		font-weight: 600;
		color: var(--text-secondary);
		margin: 10px 0 4px;
	}

	.review-text {
		margin: 6px 0;
	}

	.review-bullet {
		display: flex;
		gap: 8px;
		padding: 3px 0;
		align-items: flex-start;
	}

	.bullet-dot {
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: var(--purple);
		flex-shrink: 0;
		margin-top: 7px;
	}

	.review-bullet.numbered {
		counter-increment: review-list;
	}

	.bullet-num {
		flex-shrink: 0;
		margin-top: 0;
	}

	.bullet-num::before {
		counter-increment: review-list;
	}

	.review-code-wrapper {
		position: relative;
		margin: 8px 0;
	}

	.review-code-lang {
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

	.review-code {
		background: var(--bg-dark);
		border: 1px solid var(--border);
		border-radius: var(--radius-sm);
		padding: 10px 12px;
		margin: 0;
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		line-height: 1.5;
		overflow-x: auto;
		color: var(--text-primary);
	}

	.review-code code {
		font-family: inherit;
	}

	/* Shiki highlighted lines */
	.review-code.shiki-highlighted :global(.line) {
		display: block;
	}

	:global(.inline-code) {
		background: var(--bg-dark);
		border: 1px solid var(--border-light);
		border-radius: 3px;
		padding: 1px 5px;
		font-family: 'JetBrains Mono', monospace;
		font-size: 0.9em;
		color: var(--blue);
	}
</style>

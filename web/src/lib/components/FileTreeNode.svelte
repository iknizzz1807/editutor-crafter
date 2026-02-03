<script lang="ts">
	import type { FileTreeNode } from '$lib/types.js';
	import FileTreeNodeSelf from './FileTreeNode.svelte';

	let {
		node,
		depth = 0,
		selectedPath,
		onSelect
	}: {
		node: FileTreeNode;
		depth?: number;
		selectedPath: string | null;
		onSelect: (path: string) => void;
	} = $props();

	// svelte-ignore state_referenced_locally
	let expanded = $state(depth <= 0);

	function toggle() {
		if (node.type === 'directory') {
			expanded = !expanded;
		}
	}

	function handleClick() {
		if (node.type === 'file') {
			onSelect(node.path);
		} else {
			toggle();
		}
	}
</script>

<div class="tree-node">
	<button
		class="node-row"
		class:selected={node.type === 'file' && node.path === selectedPath}
		class:directory={node.type === 'directory'}
		style="padding-left: {depth * 16 + 8}px"
		onclick={handleClick}
	>
		{#if node.type === 'directory'}
			<span class="chevron" class:open={expanded}>{expanded ? '▼' : '▶'}</span>
			<span class="icon">📁</span>
		{:else}
			<span class="chevron-spacer"></span>
			<span class="icon">📄</span>
		{/if}
		<span class="node-name" title={node.name}>{node.name}</span>
	</button>

	{#if node.type === 'directory' && expanded && node.children}
		{#each node.children as child}
			<FileTreeNodeSelf node={child} depth={depth + 1} {selectedPath} {onSelect} />
		{/each}
	{/if}
</div>

<style>
	.tree-node {
		display: flex;
		flex-direction: column;
	}

	.node-row {
		display: flex;
		align-items: center;
		gap: 4px;
		padding: 3px 8px;
		border: none;
		background: none;
		color: var(--text-secondary);
		font-family: 'JetBrains Mono', monospace;
		font-size: 12px;
		cursor: pointer;
		white-space: nowrap;
		text-align: left;
		width: 100%;
	}

	.node-row:hover {
		background: var(--bg-card-hover);
	}

	.node-row.selected {
		background: rgba(88, 166, 255, 0.15);
		color: var(--text-primary);
	}

	.node-row.directory {
		color: var(--text-primary);
		font-weight: 500;
	}

	.chevron {
		font-size: 8px;
		width: 12px;
		flex-shrink: 0;
		text-align: center;
		color: var(--text-muted);
	}

	.chevron-spacer {
		width: 12px;
		flex-shrink: 0;
	}

	.icon {
		font-size: 13px;
		flex-shrink: 0;
	}

	.node-name {
		overflow: hidden;
		text-overflow: ellipsis;
	}
</style>

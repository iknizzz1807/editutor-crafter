<script lang="ts">
	import type { FileTreeNode } from '$lib/types.js';
	import FileTreeNodeComponent from './FileTreeNode.svelte';
	import CodeViewer from './CodeViewer.svelte';

	let {
		submissionId,
		fileName
	}: {
		submissionId: number;
		fileName: string;
	} = $props();

	let open = $state(false);
	let tree = $state<FileTreeNode | null>(null);
	let fileCount = $state(0);
	let totalSize = $state(0);
	let treeLoading = $state(false);
	let treeError = $state<string | null>(null);
	let treeFetched = $state(false);

	let selectedPath = $state<string | null>(null);
	let fileContent = $state<string | null>(null);
	let fileLanguage = $state('plaintext');
	let fileSize = $state(0);
	let fileBinary = $state(false);
	let fileTruncated = $state(false);
	let fileLoading = $state(false);
	let selectedFileName = $state('');

	async function toggleOpen() {
		open = !open;
		if (open && !treeFetched) {
			await fetchTree();
		}
	}

	async function fetchTree() {
		treeLoading = true;
		treeError = null;
		try {
			const res = await fetch(`/api/submission/${submissionId}/files`);
			const data = await res.json();
			if (!res.ok) {
				treeError = data.error || 'Failed to load file tree';
				return;
			}
			tree = data.tree;
			fileCount = data.fileCount;
			totalSize = data.totalSize;
			treeFetched = true;
		} catch {
			treeError = 'Failed to connect to server';
		} finally {
			treeLoading = false;
		}
	}

	async function selectFile(path: string) {
		if (path === selectedPath) return;
		selectedPath = path;
		selectedFileName = path.split('/').pop() || path;
		fileContent = null;
		fileBinary = false;
		fileTruncated = false;
		fileLoading = true;

		try {
			const res = await fetch(`/api/submission/${submissionId}/file?path=${encodeURIComponent(path)}`);
			const data = await res.json();
			if (!res.ok) {
				fileContent = null;
				treeError = data.error || 'Failed to load file';
				return;
			}
			fileContent = data.content;
			fileLanguage = data.language || 'plaintext';
			fileSize = data.size || 0;
			fileBinary = data.binary || false;
			fileTruncated = data.truncated || false;
		} catch {
			fileContent = null;
			treeError = 'Failed to load file';
		} finally {
			fileLoading = false;
		}
	}

	function formatSize(bytes: number): string {
		if (bytes < 1024) return bytes + ' B';
		if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
		return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
	}
</script>

<div class="file-browser-wrapper">
	<button class="btn-browse" onclick={toggleOpen}>
		{open ? '▼' : '▶'} Browse Files
		{#if treeFetched}
			<span class="file-stats">{fileCount} files, {formatSize(totalSize)}</span>
		{/if}
	</button>

	{#if open}
		<div class="file-browser">
			{#if treeLoading}
				<div class="loading">Loading file tree...</div>
			{:else if treeError}
				<div class="error-msg">{treeError}</div>
			{:else if tree}
				<div class="panels">
					<div class="tree-panel">
						{#if tree.children}
							{#each tree.children as child}
								<FileTreeNodeComponent
									node={child}
									depth={0}
									{selectedPath}
									onSelect={selectFile}
								/>
							{/each}
						{/if}
					</div>
					<div class="code-panel">
						{#if fileLoading}
							<div class="loading">Loading file...</div>
						{:else}
							<CodeViewer
								content={fileContent}
								language={fileLanguage}
								fileName={selectedFileName}
								size={fileSize}
								binary={fileBinary}
								truncated={fileTruncated}
							/>
						{/if}
					</div>
				</div>
			{/if}
		</div>
	{/if}
</div>

<style>
	.file-browser-wrapper {
		margin-top: 8px;
	}

	.btn-browse {
		display: flex;
		align-items: center;
		gap: 6px;
		padding: 4px 10px;
		background: rgba(88, 166, 255, 0.08);
		border: 1px solid rgba(88, 166, 255, 0.25);
		border-radius: 4px;
		color: var(--blue);
		cursor: pointer;
		font-size: 12px;
		font-family: inherit;
	}

	.btn-browse:hover {
		background: rgba(88, 166, 255, 0.15);
	}

	.file-stats {
		color: var(--text-muted);
		font-size: 11px;
		margin-left: 4px;
	}

	.file-browser {
		margin-top: 8px;
		border: 1px solid var(--border);
		border-radius: 6px;
		overflow: hidden;
	}

	.panels {
		display: flex;
		height: 400px;
	}

	.tree-panel {
		width: 260px;
		min-width: 200px;
		background: var(--bg-sidebar);
		border-right: 1px solid var(--border);
		overflow: auto;
		padding: 4px 0;
	}

	.code-panel {
		flex: 1;
		background: var(--bg-dark);
		overflow: hidden;
		display: flex;
		flex-direction: column;
	}

	.loading {
		display: flex;
		align-items: center;
		justify-content: center;
		padding: 32px 16px;
		color: var(--text-muted);
		font-size: 13px;
	}

	.error-msg {
		padding: 12px 16px;
		background: rgba(248, 81, 73, 0.1);
		color: var(--red);
		font-size: 13px;
	}

	@media (max-width: 640px) {
		.panels {
			flex-direction: column;
			height: 500px;
		}
		.tree-panel {
			width: 100%;
			min-width: unset;
			height: 180px;
			border-right: none;
			border-bottom: 1px solid var(--border);
		}
	}
</style>

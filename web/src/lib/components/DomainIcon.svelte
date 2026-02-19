<script lang="ts">
	import { Globe, Cpu, Database, Cloud, Brain, Gamepad2, FileCode, Shield, Wrench, CheckCircle, Trophy, Folder } from 'lucide-svelte';

	let { icon, size = 24 }: { icon: string | null | undefined, size?: number } = $props();
</script>

<div class="icon-wrapper">
	{#if icon === 'lucide:globe' || icon === 'lucide:app-window'}
		<Globe size={size} />
	{:else if icon === 'lucide:cpu'}
		<Cpu size={size} />
	{:else if icon === 'lucide:database'}
		<Database size={size} />
	{:else if icon === 'lucide:network' || icon === 'lucide:cloud'}
		<Cloud size={size} />
	{:else if icon === 'lucide:brain'}
		<Brain size={size} />
	{:else if icon === 'lucide:gamepad-2'}
		<Gamepad2 size={size} />
	{:else if icon === 'lucide:code-2' || icon === 'lucide:file-code'}
		<FileCode size={size} />
	{:else if icon === 'lucide:shield'}
		<Shield size={size} />
	{:else if icon === 'lucide:sparkles' || icon === 'lucide:wrench'}
		<Wrench size={size} />
	{:else if icon === 'lucide:git-branch' || icon === 'lucide:check-circle'}
		<CheckCircle size={size} />
	{:else if icon === 'lucide:trophy'}
		<Trophy size={size} />
	{:else}
		<Folder size={size} />
	{/if}
</div>

<style>
	.icon-wrapper {
		display: flex;
		align-items: center;
		justify-content: center;
		color: var(--blue, #3b82f6);
	}
</style>

<script lang="ts">
	import { Globe, Cpu, Database, Cloud, Brain, Gamepad2, FileCode, Shield, Wrench, CheckCircle, Trophy, Folder } from 'lucide-svelte';

	let { icon }: { icon: string | null | undefined } = $props();
</script>

<div class="icon-wrapper">
	{#if icon === 'lucide:globe' || icon === 'lucide:app-window'}
		<Globe size={24} />
	{:else if icon === 'lucide:cpu'}
		<Cpu size={24} />
	{:else if icon === 'lucide:database'}
		<Database size={24} />
	{:else if icon === 'lucide:network' || icon === 'lucide:cloud'}
		<Cloud size={24} />
	{:else if icon === 'lucide:brain'}
		<Brain size={24} />
	{:else if icon === 'lucide:gamepad-2'}
		<Gamepad2 size={24} />
	{:else if icon === 'lucide:code-2' || icon === 'lucide:file-code'}
		<FileCode size={24} />
	{:else if icon === 'lucide:shield'}
		<Shield size={24} />
	{:else if icon === 'lucide:sparkles' || icon === 'lucide:wrench'}
		<Wrench size={24} />
	{:else if icon === 'lucide:git-branch' || icon === 'lucide:check-circle'}
		<CheckCircle size={24} />
	{:else if icon === 'lucide:trophy'}
		<Trophy size={24} />
	{:else}
		<Folder size={24} />
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

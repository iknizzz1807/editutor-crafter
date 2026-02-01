<script lang="ts">
	let {
		completed,
		total,
		size = 32
	}: {
		completed: number;
		total: number;
		size?: number;
	} = $props();

	let radius = $derived(size / 2 - 4);
	let circumference = $derived(2 * Math.PI * radius);
	let pct = $derived(total > 0 ? (completed / total) * 100 : 0);
	let offset = $derived(circumference - (pct / 100) * circumference);
	let center = $derived(size / 2);
</script>

<div class="progress-circle" style="width: {size}px; height: {size}px;">
	<svg width={size} height={size} viewBox="0 0 {size} {size}">
		<circle class="bg" cx={center} cy={center} r={radius} />
		<circle
			class="fill"
			cx={center}
			cy={center}
			r={radius}
			stroke-dasharray={circumference}
			stroke-dashoffset={offset}
		/>
	</svg>
</div>

<style>
	.progress-circle {
		position: relative;
	}

	.progress-circle svg {
		transform: rotate(-90deg);
	}

	.progress-circle circle {
		fill: none;
		stroke-width: 3;
	}

	.progress-circle .bg {
		stroke: var(--bg-card);
	}

	.progress-circle .fill {
		stroke: var(--accent-bright);
		stroke-linecap: round;
		transition: stroke-dashoffset 0.5s ease;
	}
</style>

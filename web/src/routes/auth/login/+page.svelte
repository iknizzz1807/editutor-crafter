<script lang="ts">
	import { enhance } from '$app/forms';
	import AuthForm from '$lib/components/AuthForm.svelte';

	let { form } = $props();
</script>

<svelte:head>
	<title>Login - EduTutor Crafter</title>
</svelte:head>

<AuthForm mode="login" error={form?.error} username={form?.username} />

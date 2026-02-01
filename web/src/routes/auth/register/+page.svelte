<script lang="ts">
	import AuthForm from '$lib/components/AuthForm.svelte';

	let { form } = $props();
</script>

<svelte:head>
	<title>Register - EduTutor Crafter</title>
</svelte:head>

<AuthForm mode="register" error={form?.error} username={form?.username} email={form?.email} />

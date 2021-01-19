<script>
  import ImageUpload from './ImageUpload.svelte';
  import Loading from './Loading.svelte';

  let recognitionPromise;

  async function handleUpload(event) {
    const file = event.detail.file;
    const formData = new FormData();
    formData.append('image', file, file.name);

    recognitionPromise = fetch('http://localhost:8000/recognize-face/', {
      method: 'POST',
      header: { 'Content-Type': 'multipart/form-data' },
      body: formData,
    });
  }
</script>

{#if !recognitionPromise}
  <ImageUpload on:upload={handleUpload} />
{:else}
  {#await recognitionPromise}
    <Loading />
  {:then recognitionResponse}
    {#await recognitionResponse.json() then recognitionResult}
      {JSON.stringify(recognitionResult)}
    {/await}
  {:catch error}
    <p style="color: red">{error.message}</p>
  {/await}
{/if}

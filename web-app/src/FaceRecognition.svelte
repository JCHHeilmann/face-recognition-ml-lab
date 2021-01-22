<script>
  import ImageUpload from './ImageUpload.svelte';
  import Loading from './Loading.svelte';
  import Result from './Result.svelte';

  let recognitionPromise;

  async function handleUpload(event) {
    const file = event.detail.file;
    const formData = new FormData();
    formData.append('image_data', file, file.name);

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
      <Result resultData={JSON.stringify(recognitionResult).result} />
    {/await}
  {:catch error}
    <p style="color: red">{error.message}</p>
  {/await}
{/if}

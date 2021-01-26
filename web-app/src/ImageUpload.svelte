<script>
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  let file = undefined;
  let inputValue = '';

  function onFileInput({ target }) {
    file = target.files[0];
  }

  async function triggerUpload() {
    dispatch('upload', { file });
  }
</script>

<h2>Image Input</h2>

<label for="formFile" class="form-label">Pick an image of a person to recognise</label>
<input class="form-control" type="file" id="formFile" bind:value={inputValue} on:input={onFileInput} />

<button type="button" class="btn btn-primary" disabled={!file} on:click={triggerUpload}>Upload</button>

<script>
  import { createEventDispatcher } from 'svelte';

  const dispatch = createEventDispatcher();

  let file = undefined;
  let inputValue = '';

  function onFileInput({ target }) {
    file = target.files[0];

    console.log(file);
  }

  async function triggerUpload() {
    dispatch('upload', { file });
  }
</script>

<div>
  <label for="formFile" class="form-label">Pick an image of a person to recognise</label>
  <input class="form-control" type="file" id="formFile" bind:value={inputValue} on:input={onFileInput} />

  <button type="button" class="btn btn-primary" disabled={!file} on:click={triggerUpload}>Upload</button>
</div>

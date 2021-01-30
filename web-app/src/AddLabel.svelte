<script>
  import { imageStore } from './stores.js';

  let label = '';

  function addLabel() {
    const imageFile = $imageStore;

    const formData = new FormData();
    formData.append('image_data', imageFile, imageFile.name);
    formData.append('label', label);

    fetch('http://localhost:8000/add-label/', {
      method: 'POST',
      header: { 'Content-Type': 'multipart/form-data' },
      body: formData,
    });

    label = '';
  }
</script>

<h2>Person was not recognised</h2>
<label for="label-input" class="form-label">Add a label so they can be in the future</label>
<input type="text" class="form-control" id="label" bind:value={label} />
<button type="button" class="btn btn-primary" disabled={!label} on:click={addLabel}>Add</button>

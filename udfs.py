def sort_documents_by_id(data):
    # Extract the documents and ids from the data dictionary
    documents = data.get('documents', [])
    ids = data.get('ids', [])

    # Check if documents and ids are non-empty lists
    if documents and ids:
        # Zip the documents and ids together
        zipped_data = zip(documents[0], ids[0])
        # Sort the zipped data based on the ids
        sorted_data = sorted(zipped_data, key=lambda x: int(x[1]))
        # Unzip the sorted data
        sorted_documents, sorted_ids = zip(*sorted_data)
        return list(sorted_documents), list(sorted_ids)
    else:
        return [], []
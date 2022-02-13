function selectionSort(items) {
	var length = items.length;
	var lowestValueIndex = null;
	for (var a = 0; a < length; a++) {
		lowestValueIndex = a; 
		for (var b = a + 1; b < length; b++) {
			if (items[b] < items[lowestValueIndex]) {
				lowestValueIndex = b;
			}
		}
		swap(items, a, lowestValueIndex);
	}
	return items;
};
function insertionSort(items) {
	var length = items.length;
	var currentValue = undefined;
	for(var index = 1; index < length; index++) {
		currentValue = items[index];
		var position = index;
		while(position > 0) {
			var comparisonValue = items[position - 1]
			if(comparisonValue <= currentValue) {
				break;
			}
			items[position] = comparisonValue
			position--
		}
		items[position] = currentValue
	}
    return items	
}
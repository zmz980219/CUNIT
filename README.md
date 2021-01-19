1. data preprocessing Done
	1) people detection
	2) use detection result to get cloth mask
	3) store them in .npy and png file
 
2. code rewriting
	1) change dataset.py Done
	2) change model.py 
		2.1) forward Done
		2.2) add new networks parts
		2.3) update_D update_G need to be rethink and rewrite
	3) do testing and fix bugs
	
TODO list:
change the processing of mask, it should be 0 or 255(I can choose to reprocess them in dataset.py)

// Methods for filling the vectors from NumPy arrays

template<typename Type>
int fill_vec_1D(Type *arr, int *shape, vector<Type> *v)
{
	v->resize(shape[0]);
	for(int i=0; i<shape[0]; ++i)
	{	
			(*v)[i] = arr[i];
	}

	return 0;
}

template<typename Type>
int fill_vec_2D(Type *arr, int *shape, vector<vector<Type>> *v)
{
	v->resize(shape[0], vector<Type>(shape[1]));
	for(int i=0; i<shape[0]; ++i)
	{	
		for(int j=0; j<shape[1]; ++j)
			(*v)[i][j] = arr[i*shape[1]+j];
	}

	return 0;
}

template<typename Type>
int fill_vec_3D(Type *arr, int *shape, vector<vector<vector<Type>>> *v)
{
	v->resize(shape[0], vector<vector<Type>>(shape[1], vector<Type>(shape[2])));
	for(int i=0; i<shape[0]; ++i)
	{	
		for(int j=0; j<shape[1]; ++j)
		{
			for(int k=0; k<shape[2]; ++k)
				(*v)[i][j][k] = arr[i*shape[1]*shape[2]+j*shape[2]+k];
		}
	}

	return 0;
}


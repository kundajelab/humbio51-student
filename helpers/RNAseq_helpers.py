#helper scripts
def load_rnaseq_data(systems_subset,
                     rnaseq_data,
                     metadata):
    import pandas as pd
    data=pd.read_csv(
        filepath_or_buffer=rnaseq_data,
        header=0,
        index_col=0,
        sep='\t').transpose()
    metadata= pd.read_csv(
             filepath_or_buffer=metadata,
             header=0,
             index_col=0,
             sep='\t')
    nrow_data,ncol_data=data.shape
    nrow_metadata,ncol_metadata=data.shape 
    merged_df=pd.merge(data, metadata, left_index=True,right_index=True)
    if systems_subset=="all":
        data_filtered=merged_df.iloc[:,0:ncol_data].transpose()
        metadata_filtered=merged_df.iloc[:,0:-1*ncol_metadata]
    else:
        samples_to_keep=merged_df['System'].isin(systems_subset)
        merged_df_subset=merged_df[samples_to_keep]
        data_filtered=merged_df_subset.iloc[:,0:ncol_data].transpose()
        metadata_filtered=merged_df_subset.iloc[:,0:-1*ncol_metadata]
    return data_filtered,metadata_filtered


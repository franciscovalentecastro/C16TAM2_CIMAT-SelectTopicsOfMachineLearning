library(readr)

data = irosva_mx_train
concat = ''
for( elem in data$MESSAGE ){
  concat = paste(concat, elem, sep = '')
}

split = strsplit(concat, split='')
t = table(split)



cu_merge = merge(x=irosva_cu_test,y=irosva_cu_test_truth,by="ID")
cu_merge_sub = subset(cu_merge, select = c('ID','TOPIC.y','IS_IRONIC.y','MESSAGE')) 
colnames(cu_merge_sub) = c('ID','TOPIC','IS_IRONIC','MESSAGE')

write.csv(cu_merge_sub, file = "irosva_cu_test.csv",row.names=FALSE, 
          fileEncoding = "UTF-8", quote=TRUE)

mx_merge = merge(x=irosva_mx_test,y=irosva_mx_test_truth,by="ID")
mx_merge_sub = subset(mx_merge, select = c('ID','TOPIC.y','IS_IRONIC.y','MESSAGE')) 
colnames(mx_merge_sub) = c('ID','TOPIC','IS_IRONIC','MESSAGE')
write.csv(mx_merge_sub, file = "irosva_mx_test.csv",row.names=FALSE, 
          fileEncoding = "UTF-8",  quote=TRUE)

es_merge = merge(x=irosva_es_test,y=irosva_es_test_truth,by="ID")
es_merge_sub = subset(es_merge, select = c('ID','TOPIC.y','IS_IRONIC.y','MESSAGE')) 
colnames(es_merge_sub) = c('ID','TOPIC','IS_IRONIC','MESSAGE')
write.csv(es_merge_sub, file = "irosva_es_test.csv",row.names=FALSE, 
          fileEncoding = "cp866", quote=TRUE)




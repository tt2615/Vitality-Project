import torch

# Converts a numpy array to a torch tensor
class ToTensor(object):
    def __call__(self, data):
        return torch.tensor(data)
    
# class TextInputToTensor(object):
#     def __call__(self, data, index):
#         return torch.from_numpy(data)
    

# Log the value
class Log(object):
    def __call__(self, data):
        return torch.clip(torch.log10(data), min=0.)
    
# Normalize
class Normalize(object):
    def __call__(self, data):
        return 

# class TokenizeText(object):
#     def __call__(self, data):
#         tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
#         input_ids_title = []
#         attention_masks_title = []
#         for text in data['Item_Title']:
#             encoded_dict = tokenizer.encode_plus(text,
#                                                 add_special_tokens=True,
#                                                 max_length=32,
#                                                 pad_to_max_length=True,
#                                                 return_attention_mask=True,
#                                                 return_tensors='pt')
#             input_ids_title.append(encoded_dict['input_ids'])
#             attention_masks_title.append(encoded_dict['attention_mask'])
#         input_ids_title = torch.cat(input_ids_title, dim=0)
#         attention_masks_title = torch.cat(attention_masks_title, dim=0)

#         input_ids_content = []
#         attention_masks_content = []
#         for text in data['news_text']:
#             encoded_dict = tokenizer.encode_plus(text,
#                                                 add_special_tokens=True, 
#                                                 max_length=256,
#                                                 pad_to_max_length=True,
#                                                 return_attention_mask=True,
#                                                 return_tensors='pt')
#             input_ids_content.append(encoded_dict['input_ids'])
#             attention_masks_content.append(encoded_dict['attention_mask'])
#         input_ids_content = torch.cat(input_ids_content, dim=0)
#         attention_masks_content = torch.cat(attention_masks_content, dim=0)

#         return input_ids_title, attention_masks_title, input_ids_content, attention_masks_content
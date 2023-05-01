from src.pipelines.predictionpipeline import CustomData, PredictionPipeline
from src.utils import *

# data = CustomData(
#             carat=45,
#             depth = 78,
#             table = 34,
#             x = 23,
#             y = 34,
#             z = 21,
#             cut = 'Good',
#             color= 'D',
#             clarity = 'IF'
#         )

# new_data=data.get_data_as_dataframe()
# print (new_data)
# predict_pipeline = PredictionPipeline()
# print(type(predict_pipeline))
# print(predict_pipeline)
# pred=predict_pipeline.predict(new_data)

# print(pred)
# # results=round(pred[0],2) # Predicted price
# # print(results)

model = load_object(r'notebooks/notebooks/model.pkl')

print(type(model))
print(model.intercept_)
print(model.coef_)

preprocessor = load_object(r'artifacts/preprocessor.pkl')
print(preprocessor.get_params)
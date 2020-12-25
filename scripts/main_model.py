import scripts
from data_preprocess import ready_data
from train_model import fit_model
from save_model import save



x_train, x_test, y_train_cat, y_test_cat = ready_data()

neural_model = fit_model(x_train, y_train_cat)

save(neural_model)

print(neural_model.evaluate(x_test, y_test_cat))

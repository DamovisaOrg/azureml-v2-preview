import tensorflow as tf
import mlflow.tensorflow
from azureml.core import Workspace, Run
import requests
import sys

def main(repo_dispatch_key):
    mlflow.tensorflow.autolog()

    mnist = tf.keras.datasets.mnist

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10),
        ]
    )

    predictions = model(x_train[:1]).numpy()
    print(predictions)
    predictions = tf.nn.softmax(predictions).numpy()
    print(predictions)
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    print(loss_fn)

    model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])

    model.fit(x_train, y_train, epochs=5)
    model.evaluate(x_test, y_test, verbose=2)

    # get run context
    run = Run.get_context()

    # save the model
    model.save('outputs')
    print("model saved to outputs")

    # register the model
    run.upload_folder('mnist-dabrady', 'outputs')
    saved_model = run.register_model(model_name='mnist-dabrady-model', model_path='mnist-dabrady')
    print(f'Model {saved_model.name}:{saved_model.version} registered with AML: {saved_model.url}')

    # todo: fire off a repository_dispatch event
    post_headers = {'Authorization': f'Bearer: {repo_dispatch_key}', 'Accept': 'application/vnd.github.v3+json'}
    github_dispatch = requests.post('https://api.github.com/repos/DamovisaOrg/azureml-v2-preview/dispatches', data={'event_type':'model_registered', 'client_payload': {'name': saved_model.name, 'version': saved_model.version, 'url': saved_model.url }}, headers=post_headers) 
    print(github_dispatch)


if __name__ == "__main__":
    main(sys.argv[1])
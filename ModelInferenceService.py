import grpc
import onnxruntime as ort
import numpy as np
from concurrent import futures
from opentelemetry import trace, context
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
import model_inference_pb2, model_inference_pb2_grpc
import requests

# OpenTelemetry setup
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)
tracer = trace.get_tracer(__name__)

# Load ONNX model
session = ort.InferenceSession("model.onnx")
input_name = session.get_inputs()[0].name

class ModelInferenceService(model_inference_pb2_grpc.ModelInferenceServicer):
    def Predict(self, request, context):
        with tracer.start_as_current_span("model_inference") as span:
            span.set_attribute("model", "onnx")
            span.set_attribute("input_shape", str(request.input_data))
            
            # Perform inference
            input_data = np.array(request.input_data, dtype=np.float32).reshape(1, -1)
            output_data = session.run(None, {input_name: input_data})[0]
            
            # Determine model type and format output
            if len(output_data.shape) == 1 and output_data.shape[0] > 1:
                result = {"type": "classification", "predictions": output_data.tolist()}
            elif len(output_data.shape) == 2 and output_data.shape[1] > 1:
                result = {"type": "multi-class classification", "predictions": output_data.tolist()}
            elif len(output_data.shape) == 2 and output_data.shape[1] == 1:
                result = {"type": "regression", "value": output_data.flatten().tolist()}
            elif isinstance(output_data, np.ndarray) and output_data.ndim == 1:
                result = {"type": "binary classification", "prediction": int(output_data[0] > 0.5)}
            else:
                result = {"type": "unknown", "output": output_data.tolist()}
            
            # Send input/output to post-processing
            requests.post("http://post-processing-app", json={
                "input": request.input_data,
                "output": result
            })
            
            return model_inference_pb2.PredictResponse(output_data=str(result))

# gRPC server setup
def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    model_inference_pb2_grpc.add_ModelInferenceServicer_to_server(ModelInferenceService(), server)
    server.add_insecure_port("[::]:50051")
    server.start()
    server.wait_for_termination()

if __name__ == "__main__":
    serve()


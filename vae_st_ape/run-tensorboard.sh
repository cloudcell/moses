#!/bin/bash

# Run TensorBoard
echo "Starting TensorBoard in the background..."
# tensorboard --logdir ./logs_02_tensorboard --port 6006 > /dev/null 2>&1 &
tensorboard --logdir ./runs --port 6009 > /dev/null 2>&1 &
TENSORBOARD_PID=$!
echo "TensorBoard started with PID: $TENSORBOARD_PID"
echo "Access TensorBoard at http://localhost:6006"
echo "You can continue using this terminal."
echo "To stop TensorBoard later, run: kill $TENSORBOARD_PID"
echo



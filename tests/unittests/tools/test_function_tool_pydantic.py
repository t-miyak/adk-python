# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Pydantic model conversion tests

from typing import Optional
from unittest.mock import MagicMock

from google.adk.agents.invocation_context import InvocationContext
from google.adk.sessions.session import Session
from google.adk.tools.function_tool import FunctionTool
from google.adk.tools.tool_context import ToolContext
import pydantic
import pytest


class UserModel(pydantic.BaseModel):
  """Test Pydantic model for user data."""

  name: str
  age: int
  email: Optional[str] = None


class PreferencesModel(pydantic.BaseModel):
  """Test Pydantic model for preferences."""

  theme: str = "light"
  notifications: bool = True


def sync_function_with_pydantic_model(user: UserModel) -> dict:
  """Sync function that takes a Pydantic model."""
  return {
      "name": user.name,
      "age": user.age,
      "email": user.email,
      "type": str(type(user).__name__),
  }


async def async_function_with_pydantic_model(user: UserModel) -> dict:
  """Async function that takes a Pydantic model."""
  return {
      "name": user.name,
      "age": user.age,
      "email": user.email,
      "type": str(type(user).__name__),
  }


def function_with_optional_pydantic_model(
    user: UserModel, preferences: Optional[PreferencesModel] = None
) -> dict:
  """Function with required and optional Pydantic models."""
  result = {
      "user_name": user.name,
      "user_type": str(type(user).__name__),
  }
  if preferences:
    result.update({
        "theme": preferences.theme,
        "notifications": preferences.notifications,
        "preferences_type": str(type(preferences).__name__),
    })
  return result


def function_with_mixed_args(
    name: str, user: UserModel, count: int = 5
) -> dict:
  """Function with mixed argument types including Pydantic model."""
  return {
      "name": name,
      "user_name": user.name,
      "user_type": str(type(user).__name__),
      "count": count,
  }


def test_preprocess_args_with_dict_to_pydantic_conversion():
  """Test _preprocess_args converts dict to Pydantic model."""
  tool = FunctionTool(sync_function_with_pydantic_model)

  input_args = {
      "user": {"name": "Alice", "age": 30, "email": "alice@example.com"}
  }

  processed_args = tool._preprocess_args(input_args)

  # Check that the dict was converted to a Pydantic model
  assert "user" in processed_args
  user = processed_args["user"]
  assert isinstance(user, UserModel)
  assert user.name == "Alice"
  assert user.age == 30
  assert user.email == "alice@example.com"


def test_preprocess_args_with_existing_pydantic_model():
  """Test _preprocess_args leaves existing Pydantic model unchanged."""
  tool = FunctionTool(sync_function_with_pydantic_model)

  # Create an existing Pydantic model
  existing_user = UserModel(name="Bob", age=25)
  input_args = {"user": existing_user}

  processed_args = tool._preprocess_args(input_args)

  # Check that the existing model was not changed (same object)
  assert "user" in processed_args
  user = processed_args["user"]
  assert user is existing_user
  assert isinstance(user, UserModel)
  assert user.name == "Bob"


def test_preprocess_args_with_optional_pydantic_model_none():
  """Test _preprocess_args handles None for optional Pydantic models."""
  tool = FunctionTool(function_with_optional_pydantic_model)

  input_args = {"user": {"name": "Charlie", "age": 35}, "preferences": None}

  processed_args = tool._preprocess_args(input_args)

  # Check user conversion
  assert isinstance(processed_args["user"], UserModel)
  assert processed_args["user"].name == "Charlie"

  # Check preferences remains None
  assert processed_args["preferences"] is None


def test_preprocess_args_with_optional_pydantic_model_dict():
  """Test _preprocess_args converts dict for optional Pydantic models."""
  tool = FunctionTool(function_with_optional_pydantic_model)

  input_args = {
      "user": {"name": "Diana", "age": 28},
      "preferences": {"theme": "dark", "notifications": False},
  }

  processed_args = tool._preprocess_args(input_args)

  # Check both conversions
  assert isinstance(processed_args["user"], UserModel)
  assert processed_args["user"].name == "Diana"

  assert isinstance(processed_args["preferences"], PreferencesModel)
  assert processed_args["preferences"].theme == "dark"
  assert processed_args["preferences"].notifications is False


def test_preprocess_args_with_mixed_types():
  """Test _preprocess_args handles mixed argument types correctly."""
  tool = FunctionTool(function_with_mixed_args)

  input_args = {
      "name": "test_name",
      "user": {"name": "Eve", "age": 40},
      "count": 10,
  }

  processed_args = tool._preprocess_args(input_args)

  # Check that only Pydantic model was converted
  assert processed_args["name"] == "test_name"  # string unchanged
  assert processed_args["count"] == 10  # int unchanged

  # Check Pydantic model conversion
  assert isinstance(processed_args["user"], UserModel)
  assert processed_args["user"].name == "Eve"
  assert processed_args["user"].age == 40


def test_preprocess_args_with_invalid_data_graceful_failure():
  """Test _preprocess_args handles invalid data gracefully."""
  tool = FunctionTool(sync_function_with_pydantic_model)

  # Invalid data that can't be converted to UserModel
  input_args = {"user": "invalid_string"}  # string instead of dict/model

  processed_args = tool._preprocess_args(input_args)

  # Should keep original value when conversion fails
  assert processed_args["user"] == "invalid_string"


def test_preprocess_args_with_non_pydantic_parameters():
  """Test _preprocess_args ignores non-Pydantic parameters."""

  def simple_function(name: str, age: int) -> dict:
    return {"name": name, "age": age}

  tool = FunctionTool(simple_function)

  input_args = {"name": "test", "age": 25}
  processed_args = tool._preprocess_args(input_args)

  # Should remain unchanged (no Pydantic models to convert)
  assert processed_args == input_args


@pytest.mark.asyncio
async def test_run_async_with_pydantic_model_conversion_sync_function():
  """Test run_async with Pydantic model conversion for sync function."""
  tool = FunctionTool(sync_function_with_pydantic_model)

  tool_context_mock = MagicMock(spec=ToolContext)
  invocation_context_mock = MagicMock(spec=InvocationContext)
  session_mock = MagicMock(spec=Session)
  invocation_context_mock.session = session_mock
  tool_context_mock.invocation_context = invocation_context_mock

  args = {"user": {"name": "Frank", "age": 45, "email": "frank@example.com"}}

  result = await tool.run_async(args=args, tool_context=tool_context_mock)

  # Verify the function received a proper Pydantic model
  assert result["name"] == "Frank"
  assert result["age"] == 45
  assert result["email"] == "frank@example.com"
  assert result["type"] == "UserModel"


@pytest.mark.asyncio
async def test_run_async_with_pydantic_model_conversion_async_function():
  """Test run_async with Pydantic model conversion for async function."""
  tool = FunctionTool(async_function_with_pydantic_model)

  tool_context_mock = MagicMock(spec=ToolContext)
  invocation_context_mock = MagicMock(spec=InvocationContext)
  session_mock = MagicMock(spec=Session)
  invocation_context_mock.session = session_mock
  tool_context_mock.invocation_context = invocation_context_mock

  args = {"user": {"name": "Grace", "age": 32}}

  result = await tool.run_async(args=args, tool_context=tool_context_mock)

  # Verify the function received a proper Pydantic model
  assert result["name"] == "Grace"
  assert result["age"] == 32
  assert result["email"] is None  # default value
  assert result["type"] == "UserModel"


@pytest.mark.asyncio
async def test_run_async_with_optional_pydantic_models():
  """Test run_async with optional Pydantic models."""
  tool = FunctionTool(function_with_optional_pydantic_model)

  tool_context_mock = MagicMock(spec=ToolContext)
  invocation_context_mock = MagicMock(spec=InvocationContext)
  session_mock = MagicMock(spec=Session)
  invocation_context_mock.session = session_mock
  tool_context_mock.invocation_context = invocation_context_mock

  # Test with both required and optional models
  args = {
      "user": {"name": "Henry", "age": 50},
      "preferences": {"theme": "dark", "notifications": True},
  }

  result = await tool.run_async(args=args, tool_context=tool_context_mock)

  assert result["user_name"] == "Henry"
  assert result["user_type"] == "UserModel"
  assert result["theme"] == "dark"
  assert result["notifications"] is True
  assert result["preferences_type"] == "PreferencesModel"


def function_with_list_of_pydantic_models(users: list[UserModel]) -> dict:
  """Function that takes a list of Pydantic models."""
  return {
      "count": len(users),
      "names": [user.name for user in users],
      "ages": [user.age for user in users],
      "types": [type(user).__name__ for user in users],
  }


def function_with_optional_list_of_pydantic_models(
    users: Optional[list[UserModel]] = None,
) -> dict:
  """Function that takes an optional list of Pydantic models."""
  if users is None:
    return {"count": 0, "names": []}
  return {
      "count": len(users),
      "names": [user.name for user in users],
  }


def test_preprocess_args_with_list_of_dicts_to_pydantic_models():
  """Test _preprocess_args converts list of dicts to list of Pydantic models."""
  tool = FunctionTool(function_with_list_of_pydantic_models)

  input_args = {
      "users": [
          {"name": "Alice", "age": 30, "email": "alice@example.com"},
          {"name": "Bob", "age": 25},
          {"name": "Charlie", "age": 35, "email": "charlie@example.com"},
      ]
  }

  processed_args = tool._preprocess_args(input_args)

  # Check that the list of dicts was converted to a list of Pydantic models
  assert "users" in processed_args
  users = processed_args["users"]
  assert isinstance(users, list)
  assert len(users) == 3

  # Check each element is a Pydantic model with correct data
  assert isinstance(users[0], UserModel)
  assert users[0].name == "Alice"
  assert users[0].age == 30
  assert users[0].email == "alice@example.com"

  assert isinstance(users[1], UserModel)
  assert users[1].name == "Bob"
  assert users[1].age == 25
  assert users[1].email is None

  assert isinstance(users[2], UserModel)
  assert users[2].name == "Charlie"
  assert users[2].age == 35
  assert users[2].email == "charlie@example.com"


def test_preprocess_args_with_optional_list_of_pydantic_models_none():
  """Test _preprocess_args handles None for optional list parameter."""
  tool = FunctionTool(function_with_optional_list_of_pydantic_models)

  input_args = {"users": None}

  processed_args = tool._preprocess_args(input_args)

  # Check that None is preserved
  assert "users" in processed_args
  assert processed_args["users"] is None


def test_preprocess_args_with_optional_list_of_pydantic_models_with_data():
  """Test _preprocess_args converts list for optional list parameter."""
  tool = FunctionTool(function_with_optional_list_of_pydantic_models)

  input_args = {
      "users": [
          {"name": "Alice", "age": 30},
          {"name": "Bob", "age": 25},
      ]
  }

  processed_args = tool._preprocess_args(input_args)

  # Check conversion
  assert "users" in processed_args
  users = processed_args["users"]
  assert len(users) == 2
  assert all(isinstance(user, UserModel) for user in users)
  assert users[0].name == "Alice"
  assert users[1].name == "Bob"


def test_preprocess_args_with_list_keeps_invalid_items_as_original():
  """Test _preprocess_args keeps original data for items that fail validation."""
  tool = FunctionTool(function_with_list_of_pydantic_models)

  input_args = {
      "users": [
          {"name": "Alice", "age": 30},
          {"name": "Invalid"},  # Missing required 'age' field
          {"name": "Bob", "age": 25},
      ]
  }

  processed_args = tool._preprocess_args(input_args)

  # Check that all items are preserved
  assert "users" in processed_args
  users = processed_args["users"]
  assert len(users) == 3  # All items preserved

  # First item should be converted to UserModel
  assert isinstance(users[0], UserModel)
  assert users[0].name == "Alice"
  assert users[0].age == 30

  # Second item should remain as dict (failed validation)
  assert isinstance(users[1], dict)
  assert users[1] == {"name": "Invalid"}

  # Third item should be converted to UserModel
  assert isinstance(users[2], UserModel)
  assert users[2].name == "Bob"
  assert users[2].age == 25

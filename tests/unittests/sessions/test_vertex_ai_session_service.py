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

import copy
import re
import types
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from unittest import mock

from dateutil.parser import isoparse
from fastapi.openapi import models as openapi_models
from google.adk.auth import auth_schemes
from google.adk.auth.auth_tool import AuthConfig
from google.adk.events.event import Event
from google.adk.events.event_actions import EventActions
from google.adk.sessions.base_session_service import GetSessionConfig
from google.adk.sessions.session import Session
from google.adk.sessions.vertex_ai_session_service import VertexAiSessionService
from google.api_core import exceptions as api_core_exceptions
from google.genai import types as genai_types
import pytest

MOCK_SESSION_JSON_1 = {
    'name': (
        'projects/test-project/locations/test-location/'
        'reasoningEngines/123/sessions/1'
    ),
    'create_time': '2024-12-12T12:12:12.123456Z',
    'update_time': '2024-12-12T12:12:12.123456Z',
    'session_state': {
        'key': {'value': 'test_value'},
    },
    'user_id': 'user',
}
MOCK_SESSION_JSON_2 = {
    'name': (
        'projects/test-project/locations/test-location/'
        'reasoningEngines/123/sessions/2'
    ),
    'update_time': '2024-12-13T12:12:12.123456Z',
    'user_id': 'user',
}
MOCK_SESSION_JSON_3 = {
    'name': (
        'projects/test-project/locations/test-location/'
        'reasoningEngines/123/sessions/3'
    ),
    'update_time': '2024-12-14T12:12:12.123456Z',
    'user_id': 'user2',
}
MOCK_EVENT_JSON = [
    {
        'name': (
            'projects/test-project/locations/test-location/'
            'reasoningEngines/123/sessions/1/events/123'
        ),
        'invocation_id': '123',
        'author': 'user',
        'timestamp': '2024-12-12T12:12:12.123456Z',
        'content': {
            'parts': [
                {'text': 'test_content'},
            ],
        },
        'actions': {
            'state_delta': {
                'key': {'value': 'test_value'},
            },
            'transfer_agent': 'agent',
        },
        'event_metadata': {
            'partial': False,
            'turn_complete': True,
            'interrupted': False,
            'branch': '',
            'long_running_tool_ids': ['tool1'],
        },
    },
]
MOCK_EVENT_JSON_2 = [
    {
        'name': (
            'projects/test-project/locations/test-location/'
            'reasoningEngines/123/sessions/2/events/123'
        ),
        'invocation_id': '222',
        'author': 'user',
        'timestamp': '2024-12-12T12:12:12.123456Z',
    },
]
MOCK_EVENT_JSON_3 = [
    {
        'name': (
            'projects/test-project/locations/test-location/'
            'reasoningEngines/123/sessions/2/events/456'
        ),
        'invocation_id': '333',
        'author': 'user',
        'timestamp': '2024-12-12T12:12:13.123456Z',
    },
]
MOCK_SESSION_JSON_PAGE1 = {
    'name': (
        'projects/test-project/locations/test-location/'
        'reasoningEngines/123/sessions/page1'
    ),
    'update_time': '2024-12-15T12:12:12.123456Z',
    'user_id': 'user_with_pages',
}
MOCK_SESSION_JSON_PAGE2 = {
    'name': (
        'projects/test-project/locations/test-location/'
        'reasoningEngines/123/sessions/page2'
    ),
    'update_time': '2024-12-16T12:12:12.123456Z',
    'user_id': 'user_with_pages',
}

MOCK_SESSION = Session(
    app_name='123',
    user_id='user',
    id='1',
    state=MOCK_SESSION_JSON_1['session_state'],
    last_update_time=isoparse(MOCK_SESSION_JSON_1['update_time']).timestamp(),
    events=[
        Event(
            id='123',
            invocation_id='123',
            author='user',
            timestamp=isoparse(MOCK_EVENT_JSON[0]['timestamp']).timestamp(),
            content=genai_types.Content(
                parts=[genai_types.Part(text='test_content')]
            ),
            actions=EventActions(
                transfer_to_agent='agent',
                state_delta={'key': {'value': 'test_value'}},
            ),
            partial=False,
            turn_complete=True,
            interrupted=False,
            branch='',
            long_running_tool_ids={'tool1'},
        ),
    ],
)

MOCK_SESSION_2 = Session(
    app_name='123',
    user_id='user',
    id='2',
    last_update_time=isoparse(MOCK_SESSION_JSON_2['update_time']).timestamp(),
    events=[
        Event(
            id='123',
            invocation_id='222',
            author='user',
            timestamp=isoparse(MOCK_EVENT_JSON_2[0]['timestamp']).timestamp(),
        ),
        Event(
            id='456',
            invocation_id='333',
            author='user',
            timestamp=isoparse(MOCK_EVENT_JSON_3[0]['timestamp']).timestamp(),
        ),
    ],
)


class PydanticNamespace(types.SimpleNamespace):

  def model_dump(self, exclude_none=True, mode='python'):
    d = {}
    for k, v in self.__dict__.items():
      if exclude_none and v is None:
        continue
      if isinstance(v, PydanticNamespace):
        d[k] = v.model_dump(exclude_none=exclude_none, mode=mode)
      elif isinstance(v, list):
        d[k] = [
            i.model_dump(exclude_none=exclude_none, mode=mode)
            if isinstance(i, PydanticNamespace)
            else i
            for i in v
        ]
      else:
        d[k] = v
    return d


def _convert_to_object(data):
  if isinstance(data, dict):
    kwargs = {}
    for key, value in data.items():
      if key in [
          'timestamp',
          'update_time',
          'create_time',
      ] and isinstance(value, str):
        kwargs[key] = isoparse(value)
      elif key in [
          'session_state',
          'state_delta',
          'artifact_delta',
          'custom_metadata',
          'requested_auth_configs',
      ]:
        kwargs[key] = value
      else:
        kwargs[key] = _convert_to_object(value)
    return PydanticNamespace(**kwargs)
  elif isinstance(data, list):
    return [_convert_to_object(item) for item in data]
  else:
    return data


class MockApiClient:
  """Mocks the API Client."""

  def __init__(self) -> None:
    """Initializes MockClient."""
    self.session_dict: dict[str, Any] = {}
    self.event_dict: dict[str, Tuple[List[Any], Optional[str]]] = {}
    self.agent_engines = mock.Mock()
    self.agent_engines.sessions.get.side_effect = self._get_session
    self.agent_engines.sessions.list.side_effect = self._list_sessions
    self.agent_engines.sessions.delete.side_effect = self._delete_session
    self.agent_engines.sessions.create.side_effect = self._create_session
    self.agent_engines.sessions.events.list.side_effect = self._list_events
    self.agent_engines.sessions.events.append.side_effect = self._append_event

  def _get_session(self, name: str):
    session_id = name.split('/')[-1]
    if session_id in self.session_dict:
      return _convert_to_object(self.session_dict[session_id])
    raise api_core_exceptions.NotFound(f'Session not found: {session_id}')

  def _list_sessions(self, name: str, config: dict[str, Any]):
    filter_val = config.get('filter', '')
    user_id_match = re.search(r'user_id="([^"]+)"', filter_val)
    if user_id_match:
      user_id = user_id_match.group(1)
      if user_id == 'user_with_pages':
        return [
            _convert_to_object(MOCK_SESSION_JSON_PAGE1),
            _convert_to_object(MOCK_SESSION_JSON_PAGE2),
        ]
      return [
          _convert_to_object(session)
          for session in self.session_dict.values()
          if session['user_id'] == user_id
      ]

    # No user filter, return all sessions
    return [
        _convert_to_object(session) for session in self.session_dict.values()
    ]

  def _delete_session(self, name: str):
    session_id = name.split('/')[-1]
    self.session_dict.pop(session_id)

  def _create_session(self, name: str, user_id: str, config: dict[str, Any]):
    new_session_id = '4'
    self.session_dict[new_session_id] = {
        'name': (
            'projects/test-project/locations/test-location/'
            'reasoningEngines/123/sessions/'
            + new_session_id
        ),
        'user_id': user_id,
        'session_state': config.get('session_state', {}),
        'update_time': '2024-12-12T12:12:12.123456Z',
    }
    return _convert_to_object({
        'name': (
            'projects/test_project/locations/test_location/'
            'reasoningEngines/123/sessions/'
            + new_session_id
            + '/operations/111'
        ),
        'done': True,
        'response': self.session_dict['4'],
    })

  def _list_events(self, name: str, **kwargs):
    session_id = name.split('/')[-1]
    events = []
    if session_id in self.event_dict:
      events_tuple = self.event_dict[session_id]
      events.extend(events_tuple[0])
      if events_tuple[1] == 'my_token':
        events.extend(MOCK_EVENT_JSON_3)

    config = kwargs.get('config', {})
    filter_str = config.get('filter', None)
    if filter_str:
      match = re.search(r'timestamp>="([^"]+)"', filter_str)
      if match:
        after_timestamp_str = match.group(1)
        after_timestamp = isoparse(after_timestamp_str)
        events = [
            event
            for event in events
            if isoparse(event['timestamp']) >= after_timestamp
        ]
    return [_convert_to_object(event) for event in events]

  def _append_event(
      self,
      name: str,
      author: str,
      invocation_id: str,
      timestamp: Any,
      config: dict[str, Any],
  ):
    session_id = name.split('/')[-1]
    event_list, token = self.event_dict.get(session_id, ([], None))
    event_id = str(len(event_list) + 1000)  # generate unique ID

    event_timestamp_str = timestamp.isoformat().replace('+00:00', 'Z')
    event_json = {
        'name': f'{name}/events/{event_id}',
        'invocation_id': invocation_id,
        'author': author,
        'timestamp': event_timestamp_str,
    }
    event_json.update(config)

    if session_id in self.session_dict:
      self.session_dict[session_id]['update_time'] = event_timestamp_str

    if session_id in self.event_dict:
      self.event_dict[session_id][0].append(event_json)
    else:
      self.event_dict[session_id] = ([event_json], None)


def mock_vertex_ai_session_service(agent_engine_id: Optional[str] = None):
  """Creates a mock Vertex AI Session service for testing."""
  return VertexAiSessionService(
      project='test-project',
      location='test-location',
      agent_engine_id=agent_engine_id,
  )


@pytest.fixture
def mock_get_api_client():
  api_client = MockApiClient()
  api_client.session_dict = {
      '1': MOCK_SESSION_JSON_1,
      '2': MOCK_SESSION_JSON_2,
      '3': MOCK_SESSION_JSON_3,
      'page1': MOCK_SESSION_JSON_PAGE1,
      'page2': MOCK_SESSION_JSON_PAGE2,
  }
  api_client.event_dict = {
      '1': (copy.deepcopy(MOCK_EVENT_JSON), None),
      '2': (copy.deepcopy(MOCK_EVENT_JSON_2), 'my_token'),
  }
  with mock.patch(
      'google.adk.sessions.vertex_ai_session_service.VertexAiSessionService._get_api_client',
      return_value=api_client,
  ):
    yield


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
@pytest.mark.parametrize('agent_engine_id', [None, '123'])
async def test_get_empty_session(agent_engine_id):
  session_service = mock_vertex_ai_session_service(agent_engine_id)
  with pytest.raises(api_core_exceptions.NotFound) as excinfo:
    await session_service.get_session(
        app_name='123', user_id='user', session_id='0'
    )
  assert str(excinfo.value) == '404 Session not found: 0'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
@pytest.mark.parametrize('agent_engine_id', [None, '123'])
async def test_get_another_user_session(agent_engine_id):
  session_service = mock_vertex_ai_session_service(agent_engine_id)
  with pytest.raises(ValueError) as excinfo:
    await session_service.get_session(
        app_name='123', user_id='user2', session_id='1'
    )
  assert str(excinfo.value) == 'Session 1 does not belong to user user2.'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_get_and_delete_session():
  session_service = mock_vertex_ai_session_service()

  assert (
      await session_service.get_session(
          app_name='123', user_id='user', session_id='1'
      )
      == MOCK_SESSION
  )

  await session_service.delete_session(
      app_name='123', user_id='user', session_id='1'
  )
  with pytest.raises(api_core_exceptions.NotFound) as excinfo:
    await session_service.get_session(
        app_name='123', user_id='user', session_id='1'
    )
  assert str(excinfo.value) == '404 Session not found: 1'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_get_session_with_page_token():
  session_service = mock_vertex_ai_session_service()

  assert (
      await session_service.get_session(
          app_name='123', user_id='user', session_id='2'
      )
      == MOCK_SESSION_2
  )


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_get_session_with_after_timestamp_filter():
  session_service = mock_vertex_ai_session_service()
  session = await session_service.get_session(
      app_name='123',
      user_id='user',
      session_id='2',
      config=GetSessionConfig(
          after_timestamp=isoparse('2024-12-12T12:12:13.0Z').timestamp()
      ),
  )
  assert session is not None
  assert len(session.events) == 1
  assert session.events[0].id == '456'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_list_sessions():
  session_service = mock_vertex_ai_session_service()
  sessions = await session_service.list_sessions(app_name='123', user_id='user')
  assert len(sessions.sessions) == 2
  assert sessions.sessions[0].id == '1'
  assert sessions.sessions[1].id == '2'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_list_sessions_with_pagination():
  session_service = mock_vertex_ai_session_service()
  sessions = await session_service.list_sessions(
      app_name='123', user_id='user_with_pages'
  )
  assert len(sessions.sessions) == 2
  assert sessions.sessions[0].id == 'page1'
  assert sessions.sessions[1].id == 'page2'


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_list_sessions_all_users():
  session_service = mock_vertex_ai_session_service()
  sessions = await session_service.list_sessions(app_name='123', user_id=None)
  assert len(sessions.sessions) == 5
  assert {s.id for s in sessions.sessions} == {'1', '2', '3', 'page1', 'page2'}


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_create_session():
  session_service = mock_vertex_ai_session_service()

  state = {'key': 'value'}
  session = await session_service.create_session(
      app_name='123', user_id='user', state=state
  )
  assert session.state == state
  assert session.app_name == '123'
  assert session.user_id == 'user'
  assert session.last_update_time is not None

  session_id = session.id
  assert session == await session_service.get_session(
      app_name='123', user_id='user', session_id=session_id
  )


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_create_session_with_custom_session_id():
  session_service = mock_vertex_ai_session_service()

  with pytest.raises(ValueError) as excinfo:
    await session_service.create_session(
        app_name='123', user_id='user', session_id='1'
    )
  assert str(excinfo.value) == (
      'User-provided Session id is not supported for VertexAISessionService.'
  )


@pytest.mark.asyncio
@pytest.mark.usefixtures('mock_get_api_client')
async def test_append_event():
  session_service = mock_vertex_ai_session_service()
  session_before_append = await session_service.get_session(
      app_name='123', user_id='user', session_id='1'
  )
  event_to_append = Event(
      invocation_id='new_invocation',
      author='model',
      timestamp=1734005533.0,
      content=genai_types.Content(parts=[genai_types.Part(text='new_content')]),
      actions=EventActions(
          transfer_to_agent='another_agent',
          state_delta={'new_key': 'new_value'},
          skip_summarization=True,
          requested_auth_configs={
              'test_auth': AuthConfig(
                  auth_scheme=auth_schemes.OAuth2(
                      flows=openapi_models.OAuthFlows(
                          implicit=openapi_models.OAuthFlowImplicit(
                              authorizationUrl='http://test.com/auth',
                              scopes={},
                          )
                      )
                  ),
              ),
          },
      ),
      error_code='1',
      error_message='test_error',
      branch='test_branch',
      custom_metadata={'custom': 'data'},
      long_running_tool_ids={'tool2'},
  )

  await session_service.append_event(session_before_append, event_to_append)

  retrieved_session = await session_service.get_session(
      app_name='123', user_id='user', session_id='1'
  )

  assert len(retrieved_session.events) == 2
  event_to_append.id = retrieved_session.events[1].id
  assert retrieved_session.events[1] == event_to_append

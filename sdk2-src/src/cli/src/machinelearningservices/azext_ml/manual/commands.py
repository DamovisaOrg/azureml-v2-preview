# ---------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# ---------------------------------------------------------

from azure.cli.core.commands import CliCommandType


def load_command_table(self, _):
    from azext_ml.generated._client_factory import cf_ml_cl
    with self.command_group('ml model', client_factory=cf_ml_cl, is_experimental=True) as g:
        g.custom_command('create', 'ml_model_create')
        g.custom_command('show', 'ml_model_show')
        g.custom_command('list', 'ml_model_list')
        g.custom_command('delete', 'ml_model_delete')
        g.custom_command('update', 'ml_model_update')

    with self.command_group('ml environment', client_factory=cf_ml_cl, is_experimental=True) as g:
        g.custom_command('create', 'ml_environment_create')
        g.custom_command('show', 'ml_environment_show')
        g.custom_command('list', 'ml_environment_list')

    with self.command_group('ml data', client_factory=cf_ml_cl, is_experimental=True) as g:
        g.custom_command('create', 'ml_data_create')
        g.custom_command('upload', 'ml_data_upload')
        g.custom_command('show', 'ml_data_show')
        g.custom_command('list', 'ml_data_list')
        g.custom_command('delete', 'ml_data_delete')

    with self.command_group('ml code', client_factory=cf_ml_cl, is_experimental=True) as g:
        g.custom_command('create', 'ml_code_create')
        g.custom_command('show', 'ml_code_show')

    with self.command_group('ml job', client_factory=cf_ml_cl, is_experimental=True) as g:
        custom_tmpl = 'azext_ml.manual.custom.job#{}'
        custom_job = CliCommandType(operations_tmpl=custom_tmpl)
        g.custom_command('download', 'ml_job_download')
        g.custom_command('stream', 'ml_job_stream')
        g.generic_update_command('update', getter_name='ml_job_show',
                                 getter_type=custom_job, setter_name='_ml_job_update',
                                 setter_type=custom_job)

    with self.command_group('ml endpoint', client_factory=cf_ml_cl, is_experimental=True) as g:
        g.custom_command('create', 'ml_endpoint_create')
        g.custom_command('show', 'ml_endpoint_show')
        g.custom_command('delete', 'ml_endpoint_delete')
        g.custom_command('list', 'ml_endpoint_list')
        g.custom_command('list-keys', 'ml_endpoint_listkeys')
        g.custom_command('log', 'ml_endpoint_get_deployment_logs')

    with self.command_group('ml compute', client_factory=cf_ml_cl, is_experimental=True) as g:
        g.custom_command('show', 'ml_compute_show')
        g.custom_command('list', 'ml_compute_list')

    with self.command_group('ml workspace', client_factory=cf_ml_cl, is_experimental=True) as g:
        g.custom_command('show', 'ml_workspace_show')
        g.custom_command('list', 'ml_workspace_list')

    from azext_ml.generated._client_factory import cf_datastore
    ml_datastore = CliCommandType(
        operations_tmpl='azext_ml.vendored_sdks.machinelearningservices.operations._datastore_operations#DatastoreOpera'
        'tions.{}',
        client_factory=cf_datastore)
    with self.command_group('ml datastore', ml_datastore, client_factory=cf_datastore, is_experimental=True) as g:
        g.custom_command('attach-blob', 'ml_datastore_attach_blob')
        g.custom_command('detach', 'ml_datastore_detach')

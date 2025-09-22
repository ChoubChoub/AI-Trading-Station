#include <linux/module.h>
#define INCLUDE_VERMAGIC
#include <linux/build-salt.h>
#include <linux/elfnote-lto.h>
#include <linux/export-internal.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

#ifdef CONFIG_UNWINDER_ORC
#include <asm/orc_header.h>
ORC_HEADER;
#endif

BUILD_SALT;
BUILD_LTO_INFO;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(".gnu.linkonce.this_module") = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

KSYMTAB_FUNC(efrm_is_pio_enabled, "", "");
KSYMTAB_FUNC(efrm_rss_context_alloc, "", "");
KSYMTAB_FUNC(efrm_rss_context_free, "", "");
KSYMTAB_FUNC(efrm_vport_alloc, "", "");
KSYMTAB_FUNC(efrm_vport_free, "", "");
KSYMTAB_FUNC(efrm_filter_insert, "", "");
KSYMTAB_FUNC(efrm_filter_remove, "", "");
KSYMTAB_FUNC(efrm_filter_redirect, "", "");
KSYMTAB_FUNC(efrm_filter_query, "", "");
KSYMTAB_FUNC(efrm_filter_block_kernel, "", "");
KSYMTAB_FUNC(efrm_port_sniff, "", "");
KSYMTAB_FUNC(efrm_tx_port_sniff, "", "");
KSYMTAB_FUNC(efrm_nondl_register_driver, "", "");
KSYMTAB_FUNC(efrm_nondl_unregister_driver, "", "");
KSYMTAB_FUNC(efrm_register_nic_notifier, "", "");
KSYMTAB_FUNC(efrm_unregister_nic_notifier, "", "");
KSYMTAB_FUNC(efhw_nic_get_net_dev, "", "");
KSYMTAB_FUNC(efrm_vi_set_alloc, "", "");
KSYMTAB_FUNC(efrm_vi_set_release, "", "");
KSYMTAB_FUNC(efrm_vi_set_num_vis, "", "");
KSYMTAB_FUNC(efrm_vi_set_get_base, "", "");
KSYMTAB_FUNC(efrm_vi_set_get_rss_context, "", "");
KSYMTAB_FUNC(efrm_vi_set_to_resource, "", "");
KSYMTAB_FUNC(efrm_vi_set_from_resource, "", "");
KSYMTAB_FUNC(efrm_vi_set_get_pd, "", "");
KSYMTAB_FUNC(efrm_pd_stack_id_get, "", "");
KSYMTAB_FUNC(efrm_pd_exclusive_rxq_token_get, "", "");
KSYMTAB_FUNC(efrm_pd_alloc, "", "");
KSYMTAB_FUNC(efrm_pd_release, "", "");
KSYMTAB_FUNC(efrm_pd_to_resource, "", "");
KSYMTAB_FUNC(efrm_pd_from_resource, "", "");
KSYMTAB_FUNC(efrm_pd_owner_id, "", "");
KSYMTAB_FUNC(efrm_pd_set_min_align, "", "");
KSYMTAB_FUNC(efrm_pd_get_min_align, "", "");
KSYMTAB_FUNC(efrm_pd_has_vport, "", "");
KSYMTAB_FUNC(efrm_pd_get_vport_id, "", "");
KSYMTAB_FUNC(efrm_pd_vport_alloc, "", "");
KSYMTAB_FUNC(efrm_pd_dma_remap_bt, "", "");
KSYMTAB_FUNC(efrm_pd_dma_map, "", "");
KSYMTAB_FUNC(efrm_pd_dma_unmap, "", "");
KSYMTAB_FUNC(efrm_pio_realloc, "", "");
KSYMTAB_FUNC(efrm_pio_alloc, "", "");
KSYMTAB_FUNC(efrm_pio_link_vi, "", "");
KSYMTAB_FUNC(efrm_pio_unlink_vi, "", "");
KSYMTAB_FUNC(efrm_pio_release, "", "");
KSYMTAB_FUNC(efrm_pio_to_resource, "", "");
KSYMTAB_FUNC(efrm_pio_from_resource, "", "");
KSYMTAB_FUNC(efrm_pio_map_kernel, "", "");
KSYMTAB_FUNC(efrm_pio_unmap_kernel, "", "");
KSYMTAB_FUNC(efrm_pio_get_size, "", "");
KSYMTAB_FUNC(efrm_ctpio_map_kernel, "", "");
KSYMTAB_FUNC(efrm_ctpio_unmap_kernel, "", "");
KSYMTAB_FUNC(efrm_rxq_to_resource, "", "");
KSYMTAB_FUNC(efrm_rxq_from_resource, "", "");
KSYMTAB_FUNC(efrm_rxq_alloc, "", "");
KSYMTAB_FUNC(efrm_rxq_release, "", "");
KSYMTAB_FUNC(efrm_rxq_refresh, "", "");
KSYMTAB_FUNC(efrm_rxq_refresh_kernel, "", "");
KSYMTAB_FUNC(efrm_rxq_request_wakeup, "", "");
KSYMTAB_FUNC(efrm_resource_ref, "", "");
KSYMTAB_FUNC(__efrm_resource_release, "", "");
KSYMTAB_FUNC(efrm_resource_release, "", "");
KSYMTAB_FUNC(efrm_vi_set_get_vi_instance, "", "");
KSYMTAB_FUNC(efrm_vi_af_xdp_kick, "", "");
KSYMTAB_FUNC(efrm_vi_qid, "", "");
KSYMTAB_FUNC(efrm_vi_rm_evq_bytes, "", "");
KSYMTAB_FUNC(efrm_vi_n_q_entries, "", "");
KSYMTAB_FUNC(efrm_vi_resource_mark_shut_down, "", "");
KSYMTAB_FUNC(efrm_vi_q_flush, "", "");
KSYMTAB_FUNC(efrm_nic_flush_all_queues, "", "");
KSYMTAB_FUNC(efrm_vi_q_alloc_sanitize_size, "", "");
KSYMTAB_FUNC(efrm_vi_q_alloc, "", "");
KSYMTAB_FUNC(efrm_vi_resource_alloc, "", "");
KSYMTAB_FUNC(efrm_vi_resource_deferred, "", "");
KSYMTAB_FUNC(__efrm_vi_attr_init, "", "");
KSYMTAB_FUNC(efrm_vi_attr_set_pd, "", "");
KSYMTAB_FUNC(efrm_vi_attr_set_packed_stream, "", "");
KSYMTAB_FUNC(efrm_vi_attr_set_ps_buffer_size, "", "");
KSYMTAB_FUNC(efrm_vi_attr_set_instance, "", "");
KSYMTAB_FUNC(efrm_vi_attr_set_interrupt_core, "", "");
KSYMTAB_FUNC(efrm_vi_attr_set_wakeup_channel, "", "");
KSYMTAB_FUNC(efrm_vi_attr_set_want_interrupt, "", "");
KSYMTAB_FUNC(efrm_vi_attr_set_queue_types, "", "");
KSYMTAB_FUNC(efrm_vi_get_efct_shm_bytes, "", "");
KSYMTAB_FUNC(efrm_vi_alloc, "", "");
KSYMTAB_FUNC(efrm_vi_is_hw_rx_loopback_supported, "", "");
KSYMTAB_FUNC(efrm_vi_is_hw_drop_filter_supported, "", "");
KSYMTAB_FUNC(efrm_vi_q_get_size, "", "");
KSYMTAB_FUNC(efrm_vi_qs_reinit, "", "");
KSYMTAB_FUNC(efrm_vi_from_resource, "", "");
KSYMTAB_FUNC(efrm_vi_tx_alt_alloc, "", "");
KSYMTAB_FUNC(efrm_vi_tx_alt_free, "", "");
KSYMTAB_FUNC(efrm_eventq_request_wakeup, "", "");
KSYMTAB_FUNC(efrm_eventq_register_callback, "", "");
KSYMTAB_FUNC(efrm_eventq_kill_callback, "", "");
KSYMTAB_FUNC(efrm_vi_wait_nic_complete_flushes, "", "");
KSYMTAB_FUNC(efrm_vi_register_flush_callback, "", "");
KSYMTAB_FUNC(efrm_pt_flush, "", "");
KSYMTAB_FUNC(efrm_vi_resource_release, "", "");
KSYMTAB_FUNC(efrm_vi_resource_stop_callback, "", "");
KSYMTAB_FUNC(efrm_vi_resource_release_flushed, "", "");
KSYMTAB_FUNC(efrm_vi_get_mappings, "", "");
KSYMTAB_FUNC(efrm_vi_get_pd, "", "");
KSYMTAB_FUNC(efrm_vi_get_dev, "", "");
KSYMTAB_FUNC(efrm_vi_get_channel, "", "");
KSYMTAB_FUNC(efrm_vi_get_rx_error_stats, "", "");
KSYMTAB_DATA(efrm_nic_tablep, "", "");
KSYMTAB_FUNC(efrm_client_disable_post_reset, "", "");
KSYMTAB_FUNC(efrm_client_get_by_nic, "", "");
KSYMTAB_FUNC(efrm_client_get, "", "");
KSYMTAB_FUNC(efrm_client_set_callbacks, "", "");
KSYMTAB_FUNC(efrm_client_put, "", "");
KSYMTAB_FUNC(efrm_client_add_ref, "", "");
KSYMTAB_FUNC(efrm_client_get_nic, "", "");
KSYMTAB_FUNC(efrm_client_get_ifindex, "", "");
KSYMTAB_FUNC(efrm_client_accel_allowed, "", "");
KSYMTAB_FUNC(efhw_nic_find, "", "");
KSYMTAB_FUNC(efhw_nic_find_by_dev, "", "");
KSYMTAB_FUNC(efhw_nic_find_by_foo, "", "");
KSYMTAB_FUNC(oo_hugetlb_allocator_create, "", "");
KSYMTAB_FUNC(oo_hugetlb_allocator_get, "", "");
KSYMTAB_FUNC(oo_hugetlb_allocator_put, "", "");
KSYMTAB_FUNC(oo_hugetlb_page_alloc_raw, "", "");
KSYMTAB_FUNC(oo_hugetlb_page_free_raw, "", "");
KSYMTAB_FUNC(oo_hugetlb_pages_prealloc, "", "");
KSYMTAB_FUNC(oo_hugetlb_page_offset, "", "");
KSYMTAB_FUNC(oo_iobufset_kfree, "", "");
KSYMTAB_FUNC(oo_iobufset_pages_release, "", "");
KSYMTAB_FUNC(oo_iobufset_pages_alloc, "", "");
KSYMTAB_FUNC(oo_iobufset_init, "", "");
KSYMTAB_FUNC(oo_iobufset_resource_release, "", "");
KSYMTAB_FUNC(oo_iobufset_resource_alloc, "", "");
KSYMTAB_FUNC(oo_iobufset_resource_remap_bt, "", "");
KSYMTAB_DATA(efrm_syscall_table, "", "");
KSYMTAB_DATA(efrm_x64_sys_call, "", "");
KSYMTAB_FUNC(efrm_syscall_table_call, "", "");

SYMBOL_CRC(efrm_is_pio_enabled, 0x66116dcf, "");
SYMBOL_CRC(efrm_rss_context_alloc, 0x6831fc54, "");
SYMBOL_CRC(efrm_rss_context_free, 0xfc868275, "");
SYMBOL_CRC(efrm_vport_alloc, 0xbaabd1d6, "");
SYMBOL_CRC(efrm_vport_free, 0xeee3813b, "");
SYMBOL_CRC(efrm_filter_insert, 0x46944f49, "");
SYMBOL_CRC(efrm_filter_remove, 0x2e34a54b, "");
SYMBOL_CRC(efrm_filter_redirect, 0xadd6bac8, "");
SYMBOL_CRC(efrm_filter_query, 0x0e2c4063, "");
SYMBOL_CRC(efrm_filter_block_kernel, 0x187a0f0b, "");
SYMBOL_CRC(efrm_port_sniff, 0x991242e6, "");
SYMBOL_CRC(efrm_tx_port_sniff, 0xf51ea829, "");
SYMBOL_CRC(efrm_nondl_register_driver, 0x8264f899, "");
SYMBOL_CRC(efrm_nondl_unregister_driver, 0x9e011c41, "");
SYMBOL_CRC(efrm_register_nic_notifier, 0xf9e3456f, "");
SYMBOL_CRC(efrm_unregister_nic_notifier, 0x5b2432e2, "");
SYMBOL_CRC(efhw_nic_get_net_dev, 0xdc57a617, "");
SYMBOL_CRC(efrm_vi_set_alloc, 0x456e383c, "");
SYMBOL_CRC(efrm_vi_set_release, 0x5f55f065, "");
SYMBOL_CRC(efrm_vi_set_num_vis, 0x7a0264fe, "");
SYMBOL_CRC(efrm_vi_set_get_base, 0x52778088, "");
SYMBOL_CRC(efrm_vi_set_get_rss_context, 0x4e0713a8, "");
SYMBOL_CRC(efrm_vi_set_to_resource, 0x120626ef, "");
SYMBOL_CRC(efrm_vi_set_from_resource, 0xe129780e, "");
SYMBOL_CRC(efrm_vi_set_get_pd, 0x5be4da86, "");
SYMBOL_CRC(efrm_pd_stack_id_get, 0xb3de0211, "");
SYMBOL_CRC(efrm_pd_exclusive_rxq_token_get, 0xa0f2a825, "");
SYMBOL_CRC(efrm_pd_alloc, 0x026a8741, "");
SYMBOL_CRC(efrm_pd_release, 0x70fb80b3, "");
SYMBOL_CRC(efrm_pd_to_resource, 0x903ea996, "");
SYMBOL_CRC(efrm_pd_from_resource, 0xbce46fd7, "");
SYMBOL_CRC(efrm_pd_owner_id, 0x6e133dad, "");
SYMBOL_CRC(efrm_pd_set_min_align, 0x67b0a33e, "");
SYMBOL_CRC(efrm_pd_get_min_align, 0x498a374d, "");
SYMBOL_CRC(efrm_pd_has_vport, 0x758dd297, "");
SYMBOL_CRC(efrm_pd_get_vport_id, 0x740d27ca, "");
SYMBOL_CRC(efrm_pd_vport_alloc, 0xda405ec8, "");
SYMBOL_CRC(efrm_pd_dma_remap_bt, 0x755b5383, "");
SYMBOL_CRC(efrm_pd_dma_map, 0x111eb4d5, "");
SYMBOL_CRC(efrm_pd_dma_unmap, 0x2119da42, "");
SYMBOL_CRC(efrm_pio_realloc, 0x48aef43f, "");
SYMBOL_CRC(efrm_pio_alloc, 0x36986d6a, "");
SYMBOL_CRC(efrm_pio_link_vi, 0x6873b0fc, "");
SYMBOL_CRC(efrm_pio_unlink_vi, 0xeaec368f, "");
SYMBOL_CRC(efrm_pio_release, 0xe1451210, "");
SYMBOL_CRC(efrm_pio_to_resource, 0x8af99b5d, "");
SYMBOL_CRC(efrm_pio_from_resource, 0x2ddeeded, "");
SYMBOL_CRC(efrm_pio_map_kernel, 0x560a7db7, "");
SYMBOL_CRC(efrm_pio_unmap_kernel, 0x45b66232, "");
SYMBOL_CRC(efrm_pio_get_size, 0x7f10171e, "");
SYMBOL_CRC(efrm_ctpio_map_kernel, 0x9e9b925d, "");
SYMBOL_CRC(efrm_ctpio_unmap_kernel, 0x7bf2a3f0, "");
SYMBOL_CRC(efrm_rxq_to_resource, 0x340b60b4, "");
SYMBOL_CRC(efrm_rxq_from_resource, 0xc7a85e81, "");
SYMBOL_CRC(efrm_rxq_alloc, 0x42391ebf, "");
SYMBOL_CRC(efrm_rxq_release, 0x28e0c045, "");
SYMBOL_CRC(efrm_rxq_refresh, 0x90798ffc, "");
SYMBOL_CRC(efrm_rxq_refresh_kernel, 0x9ddd563b, "");
SYMBOL_CRC(efrm_rxq_request_wakeup, 0x9d8ac8eb, "");
SYMBOL_CRC(efrm_resource_ref, 0xc034847f, "");
SYMBOL_CRC(__efrm_resource_release, 0x2229cc7c, "");
SYMBOL_CRC(efrm_resource_release, 0x72b6ecc8, "");
SYMBOL_CRC(efrm_vi_set_get_vi_instance, 0xc573bfb0, "");
SYMBOL_CRC(efrm_vi_af_xdp_kick, 0xfe592b7d, "");
SYMBOL_CRC(efrm_vi_qid, 0x7566e888, "");
SYMBOL_CRC(efrm_vi_rm_evq_bytes, 0x03508850, "");
SYMBOL_CRC(efrm_vi_n_q_entries, 0x3a00b289, "");
SYMBOL_CRC(efrm_vi_resource_mark_shut_down, 0x29945efb, "");
SYMBOL_CRC(efrm_vi_q_flush, 0xff3611e0, "");
SYMBOL_CRC(efrm_nic_flush_all_queues, 0x7414e858, "");
SYMBOL_CRC(efrm_vi_q_alloc_sanitize_size, 0x0e33f6d5, "");
SYMBOL_CRC(efrm_vi_q_alloc, 0x66c77b30, "");
SYMBOL_CRC(efrm_vi_resource_alloc, 0x8213461d, "");
SYMBOL_CRC(efrm_vi_resource_deferred, 0x04135d82, "");
SYMBOL_CRC(__efrm_vi_attr_init, 0x32f8cf8c, "");
SYMBOL_CRC(efrm_vi_attr_set_pd, 0x9ad8ffb7, "");
SYMBOL_CRC(efrm_vi_attr_set_packed_stream, 0x08b0cab7, "");
SYMBOL_CRC(efrm_vi_attr_set_ps_buffer_size, 0x51fe0109, "");
SYMBOL_CRC(efrm_vi_attr_set_instance, 0x06cd6b2c, "");
SYMBOL_CRC(efrm_vi_attr_set_interrupt_core, 0x2f2a3b2f, "");
SYMBOL_CRC(efrm_vi_attr_set_wakeup_channel, 0x615e46f0, "");
SYMBOL_CRC(efrm_vi_attr_set_want_interrupt, 0xeecc911f, "");
SYMBOL_CRC(efrm_vi_attr_set_queue_types, 0xc5ab62aa, "");
SYMBOL_CRC(efrm_vi_get_efct_shm_bytes, 0x92070734, "");
SYMBOL_CRC(efrm_vi_alloc, 0x40f76b79, "");
SYMBOL_CRC(efrm_vi_is_hw_rx_loopback_supported, 0x30b16b0b, "");
SYMBOL_CRC(efrm_vi_is_hw_drop_filter_supported, 0x71290849, "");
SYMBOL_CRC(efrm_vi_q_get_size, 0x603487aa, "");
SYMBOL_CRC(efrm_vi_qs_reinit, 0xf2338023, "");
SYMBOL_CRC(efrm_vi_from_resource, 0x64d6136d, "");
SYMBOL_CRC(efrm_vi_tx_alt_alloc, 0x8828e7f1, "");
SYMBOL_CRC(efrm_vi_tx_alt_free, 0x55483179, "");
SYMBOL_CRC(efrm_eventq_request_wakeup, 0xbb14b9d0, "");
SYMBOL_CRC(efrm_eventq_register_callback, 0x561c768a, "");
SYMBOL_CRC(efrm_eventq_kill_callback, 0x708a3b32, "");
SYMBOL_CRC(efrm_vi_wait_nic_complete_flushes, 0x2849528c, "");
SYMBOL_CRC(efrm_vi_register_flush_callback, 0x6f304248, "");
SYMBOL_CRC(efrm_pt_flush, 0x57b89ff0, "");
SYMBOL_CRC(efrm_vi_resource_release, 0x43ae1bbf, "");
SYMBOL_CRC(efrm_vi_resource_stop_callback, 0x76a33a4f, "");
SYMBOL_CRC(efrm_vi_resource_release_flushed, 0x2f36764c, "");
SYMBOL_CRC(efrm_vi_get_mappings, 0xbc48d841, "");
SYMBOL_CRC(efrm_vi_get_pd, 0x33007032, "");
SYMBOL_CRC(efrm_vi_get_dev, 0xc95bba20, "");
SYMBOL_CRC(efrm_vi_get_channel, 0x30f318c9, "");
SYMBOL_CRC(efrm_vi_get_rx_error_stats, 0x4b863765, "");
SYMBOL_CRC(efrm_nic_tablep, 0x150a2ec8, "");
SYMBOL_CRC(efrm_client_disable_post_reset, 0xf956e049, "");
SYMBOL_CRC(efrm_client_get_by_nic, 0xc9654012, "");
SYMBOL_CRC(efrm_client_get, 0xafb9587e, "");
SYMBOL_CRC(efrm_client_set_callbacks, 0xeb76aad8, "");
SYMBOL_CRC(efrm_client_put, 0x20abce77, "");
SYMBOL_CRC(efrm_client_add_ref, 0xda0ac199, "");
SYMBOL_CRC(efrm_client_get_nic, 0xf622a9a5, "");
SYMBOL_CRC(efrm_client_get_ifindex, 0xffeb6732, "");
SYMBOL_CRC(efrm_client_accel_allowed, 0x5dcd46f7, "");
SYMBOL_CRC(efhw_nic_find, 0x31ecbc80, "");
SYMBOL_CRC(efhw_nic_find_by_dev, 0x30eb852f, "");
SYMBOL_CRC(efhw_nic_find_by_foo, 0x172fdfb2, "");
SYMBOL_CRC(oo_hugetlb_allocator_create, 0x04a03016, "");
SYMBOL_CRC(oo_hugetlb_allocator_get, 0x2f5162ed, "");
SYMBOL_CRC(oo_hugetlb_allocator_put, 0x7f5e6908, "");
SYMBOL_CRC(oo_hugetlb_page_alloc_raw, 0x49d3146d, "");
SYMBOL_CRC(oo_hugetlb_page_free_raw, 0x0d15e3f2, "");
SYMBOL_CRC(oo_hugetlb_pages_prealloc, 0x602fe4c4, "");
SYMBOL_CRC(oo_hugetlb_page_offset, 0xdef4ca32, "");
SYMBOL_CRC(oo_iobufset_kfree, 0xf4378c10, "");
SYMBOL_CRC(oo_iobufset_pages_release, 0x403af6cd, "");
SYMBOL_CRC(oo_iobufset_pages_alloc, 0x7699c808, "");
SYMBOL_CRC(oo_iobufset_init, 0x6095f6ed, "");
SYMBOL_CRC(oo_iobufset_resource_release, 0x6f97cbad, "");
SYMBOL_CRC(oo_iobufset_resource_alloc, 0xa73852f0, "");
SYMBOL_CRC(oo_iobufset_resource_remap_bt, 0xb25ca08d, "");
SYMBOL_CRC(efrm_syscall_table, 0xd1a86bb4, "");
SYMBOL_CRC(efrm_x64_sys_call, 0x23dea42f, "");
SYMBOL_CRC(efrm_syscall_table_call, 0x0a49ac00, "");

static const char ____versions[]
__used __section("__versions") =
	"\x14\x00\x00\x00\x32\xb4\x35\x8a"
	"sme_me_mask\0"
	"\x1c\x00\x00\x00\xd7\x22\x7f\x58"
	"devmap_managed_key\0\0"
	"\x14\x00\x00\x00\x3b\x4a\x51\xc1"
	"free_irq\0\0\0\0"
	"\x18\x00\x00\x00\xce\xb0\x1d\xc3"
	"is_vmalloc_addr\0"
	"\x14\x00\x00\x00\x2a\xc6\xdc\xc8"
	"krealloc\0\0\0\0"
	"\x18\x00\x00\x00\xb9\xda\x0c\xa7"
	"__sock_create\0\0\0"
	"\x1c\x00\x00\x00\x48\x9f\xdb\x88"
	"__check_object_size\0"
	"\x14\x00\x00\x00\xf2\x0f\x72\x6e"
	"rtnl_unlock\0"
	"\x18\x00\x00\x00\x28\x30\xfd\x23"
	"vmalloc_node\0\0\0\0"
	"\x18\x00\x00\x00\xff\xa6\x4d\x7b"
	"__init_rwsem\0\0\0\0"
	"\x18\x00\x00\x00\xed\x25\xcd\x49"
	"alloc_workqueue\0"
	"\x18\x00\x00\x00\xc2\x9c\xc4\x13"
	"_copy_from_user\0"
	"\x18\x00\x00\x00\x14\x27\x52\x8d"
	"__rcu_read_lock\0"
	"\x10\x00\x00\x00\xa0\x13\xbc\x77"
	"strim\0\0\0"
	"\x18\x00\x00\x00\x1d\x0f\x67\x85"
	"rtnl_is_locked\0\0"
	"\x14\x00\x00\x00\x78\xe6\xbd\x2d"
	"proc_create\0"
	"\x18\x00\x00\x00\x30\x99\xe7\xc3"
	"vmalloc_to_page\0"
	"\x10\x00\x00\x00\xeb\x02\xe6\xb0"
	"memmove\0"
	"\x14\x00\x00\x00\xcb\x25\xb8\x05"
	"seq_release\0"
	"\x14\x00\x00\x00\x6e\x4a\x6e\x65"
	"snprintf\0\0\0\0"
	"\x14\x00\x00\x00\x2f\x7a\x25\xa6"
	"complete\0\0\0\0"
	"\x18\x00\x00\x00\x36\xf2\xb6\xc5"
	"queue_work_on\0\0\0"
	"\x14\x00\x00\x00\x75\xc2\x9b\x5c"
	"pci_dev_put\0"
	"\x14\x00\x00\x00\x0c\x86\x56\x5b"
	"vm_munmap\0\0\0"
	"\x18\x00\x00\x00\x51\x52\xea\x21"
	"__bitmap_weight\0"
	"\x20\x00\x00\x00\xb5\x41\x87\x60"
	"__init_swait_queue_head\0"
	"\x14\x00\x00\x00\xbf\x0f\x54\x92"
	"finish_wait\0"
	"\x20\x00\x00\x00\x8d\x5e\x74\x66"
	"dma_unmap_page_attrs\0\0\0\0"
	"\x14\x00\x00\x00\x86\x81\x84\x96"
	"scnprintf\0\0\0"
	"\x30\x00\x00\x00\x8c\x95\x4a\xc8"
	"__mmap_lock_do_trace_acquire_returned\0\0\0"
	"\x24\x00\x00\x00\x6f\x6f\x23\x4c"
	"__x86_indirect_thunk_r15\0\0\0\0"
	"\x10\x00\x00\x00\x30\x46\xac\xcc"
	"fget\0\0\0\0"
	"\x14\x00\x00\x00\x9b\x84\x97\x14"
	"kernel_bind\0"
	"\x10\x00\x00\x00\x53\x39\xc0\xed"
	"iounmap\0"
	"\x1c\x00\x00\x00\xa7\x73\x46\xe5"
	"unpin_user_pages\0\0\0\0"
	"\x14\x00\x00\x00\x1a\xb9\x05\xca"
	"fd_install\0\0"
	"\x10\x00\x00\x00\x38\xdf\xac\x69"
	"memcpy\0\0"
	"\x10\x00\x00\x00\x83\x12\x96\x94"
	"vunmap\0\0"
	"\x10\x00\x00\x00\xba\x0c\x7a\x03"
	"kfree\0\0\0"
	"\x14\x00\x00\x00\x22\xf7\xaf\xb8"
	"pcpu_hot\0\0\0\0"
	"\x14\x00\x00\x00\x12\xf2\x69\x73"
	"seq_lseek\0\0\0"
	"\x18\x00\x00\x00\x38\x22\xfb\x4a"
	"add_wait_queue\0\0"
	"\x28\x00\x00\x00\xa7\xdf\x92\x72"
	"__put_devmap_managed_page_refs\0\0"
	"\x10\x00\x00\x00\x1e\x5d\x7f\x9c"
	"vm_mmap\0"
	"\x1c\x00\x00\x00\x83\x71\xc6\xdf"
	"proc_create_data\0\0\0\0"
	"\x20\x00\x00\x00\x95\xd4\x26\x8c"
	"prepare_to_wait_event\0\0\0"
	"\x30\x00\x00\x00\x03\xf8\xde\x2e"
	"__tracepoint_mmap_lock_acquire_returned\0"
	"\x14\x00\x00\x00\x44\x43\x96\xe2"
	"__wake_up\0\0\0"
	"\x14\x00\x00\x00\xeb\xd0\x02\x43"
	"free_pages\0\0"
	"\x14\x00\x00\x00\x7d\x47\xde\x17"
	"get_device\0\0"
	"\x18\x00\x00\x00\x4b\xf6\xd5\x48"
	"sock_alloc_file\0"
	"\x28\x00\x00\x00\x86\xcf\x10\xc1"
	"__tracepoint_mmap_lock_released\0"
	"\x18\x00\x00\x00\x64\xbd\x8f\xba"
	"_raw_spin_lock\0\0"
	"\x18\x00\x00\x00\x61\x12\xd5\x98"
	"vfs_truncate\0\0\0\0"
	"\x18\x00\x00\x00\x8c\x89\xd4\xcb"
	"fortify_panic\0\0\0"
	"\x14\x00\x00\x00\xbb\x6d\xfb\xbd"
	"__fentry__\0\0"
	"\x20\x00\x00\x00\xaf\xb1\x31\x2c"
	"register_pernet_subsys\0\0"
	"\x1c\x00\x00\x00\xde\x76\x52\x1e"
	"dev_driver_string\0\0\0"
	"\x18\x00\x00\x00\x27\x97\x6d\x3a"
	"pin_user_pages\0\0"
	"\x24\x00\x00\x00\x97\x70\x48\x65"
	"__x86_indirect_thunk_rax\0\0\0\0"
	"\x24\x00\x00\x00\xba\xe1\x32\xfd"
	"auxiliary_driver_unregister\0"
	"\x1c\x00\x00\x00\xe3\x41\xf0\xbb"
	"dma_map_page_attrs\0\0"
	"\x1c\x00\x00\x00\x16\x31\xe7\xb5"
	"flush_delayed_work\0\0"
	"\x10\x00\x00\x00\x7e\x3a\x2c\x12"
	"_printk\0"
	"\x18\x00\x00\x00\x37\xd4\x9d\x3d"
	"vfs_fallocate\0\0\0"
	"\x14\x00\x00\x00\x51\x0e\x00\x01"
	"schedule\0\0\0\0"
	"\x1c\x00\x00\x00\xcb\xf6\xfd\xf0"
	"__stack_chk_fail\0\0\0\0"
	"\x18\x00\x00\x00\x0a\xa6\x35\x56"
	"vmalloc_user\0\0\0\0"
	"\x20\x00\x00\x00\x5f\x69\x96\x02"
	"refcount_warn_saturate\0\0"
	"\x20\x00\x00\x00\x6d\xb5\xfc\xb2"
	"queue_delayed_work_on\0\0\0"
	"\x1c\x00\x00\x00\xca\x21\x60\xe4"
	"_raw_spin_unlock_bh\0"
	"\x24\x00\x00\x00\xbb\x58\xed\x11"
	"debugfs_lookup_and_remove\0\0\0"
	"\x14\x00\x00\x00\x99\xf0\xa0\x42"
	"put_device\0\0"
	"\x10\x00\x00\x00\x94\xb6\x16\xa9"
	"strnlen\0"
	"\x14\x00\x00\x00\xfc\x11\x89\x61"
	"numa_node\0\0\0"
	"\x24\x00\x00\x00\x7c\xb2\x83\x63"
	"__x86_indirect_thunk_rdx\0\0\0\0"
	"\x18\x00\x00\x00\xad\x58\x65\x8c"
	"__free_pages\0\0\0\0"
	"\x10\x00\x00\x00\x49\xb3\xa9\x40"
	"vzalloc\0"
	"\x10\x00\x00\x00\x89\xbc\xcb\xc6"
	"capable\0"
	"\x14\x00\x00\x00\x8a\xe5\xf4\xd4"
	"find_vma\0\0\0\0"
	"\x28\x00\x00\x00\xb3\x1c\xa2\x87"
	"__ubsan_handle_out_of_bounds\0\0\0\0"
	"\x1c\x00\x00\x00\x5e\xd7\xd8\x7c"
	"page_offset_base\0\0\0\0"
	"\x28\x00\x00\x00\xe4\x6f\xb3\xbc"
	"hugetlb_optimize_vmemmap_key\0\0\0\0"
	"\x20\x00\x00\x00\x39\x9d\xbc\x69"
	"bpf_prog_get_type_dev\0\0\0"
	"\x14\x00\x00\x00\x09\xe9\xfd\xb6"
	"close_fd\0\0\0\0"
	"\x18\x00\x00\x00\x75\x79\x48\xfe"
	"init_wait_entry\0"
	"\x10\x00\x00\x00\xa8\x6c\x2f\x54"
	"fput\0\0\0\0"
	"\x14\x00\x00\x00\xd2\x19\xbc\x57"
	"down_write\0\0"
	"\x14\x00\x00\x00\x08\xb5\x75\x8c"
	"init_net\0\0\0\0"
	"\x14\x00\x00\x00\x25\x7a\x80\xce"
	"up_write\0\0\0\0"
	"\x18\x00\x00\x00\x7b\xf7\x19\x1d"
	"physical_mask\0\0\0"
	"\x20\x00\x00\x00\x8e\x83\xd5\x92"
	"request_threaded_irq\0\0\0\0"
	"\x10\x00\x00\x00\xf0\x37\x7a\xa0"
	"memchr\0\0"
	"\x1c\x00\x00\x00\x0f\x81\x69\x24"
	"__rcu_read_unlock\0\0\0"
	"\x24\x00\x00\x00\x2e\x5e\x38\x55"
	"__x86_indirect_thunk_r14\0\0\0\0"
	"\x1c\x00\x00\x00\xc4\xdc\xee\xa7"
	"call_usermodehelper\0"
	"\x1c\x00\x00\x00\xb7\xf5\x18\x64"
	"dev_get_by_index\0\0\0\0"
	"\x1c\x00\x00\x00\x63\xa5\x03\x4c"
	"random_kmalloc_seed\0"
	"\x1c\x00\x00\x00\x0c\xd2\x03\x8c"
	"destroy_workqueue\0\0\0"
	"\x14\x00\x00\x00\x4b\x8d\xfa\x4d"
	"mutex_lock\0\0"
	"\x18\x00\x00\x00\xb8\x92\x7a\x6a"
	"debugfs_remove\0\0"
	"\x10\x00\x00\x00\x11\x13\x92\x5a"
	"strncmp\0"
	"\x24\x00\x00\x00\xe9\xc8\x79\x1a"
	"__x86_indirect_thunk_r13\0\0\0\0"
	"\x2c\x00\x00\x00\xe2\xcc\x3b\x2e"
	"wait_for_completion_interruptible\0\0\0"
	"\x14\x00\x00\x00\xb0\x28\x9d\x4c"
	"phys_base\0\0\0"
	"\x10\x00\x00\x00\x75\x95\x24\xdf"
	"vmap\0\0\0\0"
	"\x10\x00\x00\x00\x09\xcd\x80\xde"
	"ioremap\0"
	"\x10\x00\x00\x00\xa7\xd0\x9a\x44"
	"memcmp\0\0"
	"\x18\x00\x00\x00\x20\x2e\xd1\x9e"
	"kmalloc_large\0\0\0"
	"\x1c\x00\x00\x00\x85\x6a\x21\xe7"
	"sysfs_create_group\0\0"
	"\x10\x00\x00\x00\xe6\x6e\xab\xbc"
	"sscanf\0\0"
	"\x18\x00\x00\x00\x9f\x0c\xfb\xce"
	"__mutex_init\0\0\0\0"
	"\x10\x00\x00\x00\xc7\x9a\x08\x11"
	"_ctype\0\0"
	"\x14\x00\x00\x00\x4d\xad\x4b\x12"
	"kstrtobool\0\0"
	"\x18\x00\x00\x00\x99\x2b\x5a\x9d"
	"default_llseek\0\0"
	"\x14\x00\x00\x00\x50\x14\x15\x81"
	"proc_mkdir\0\0"
	"\x24\x00\x00\x00\x61\xd2\x8c\xd3"
	"__default_kernel_pte_mask\0\0\0"
	"\x1c\x00\x00\x00\x78\x99\xad\x3d"
	"cancel_delayed_work\0"
	"\x24\x00\x00\x00\xc0\xdd\x60\xae"
	"unregister_pernet_subsys\0\0\0\0"
	"\x1c\x00\x00\x00\x75\x3f\x68\x9e"
	"__cpu_possible_mask\0"
	"\x10\x00\x00\x00\xc5\x8f\x57\xfb"
	"memset\0\0"
	"\x24\x00\x00\x00\x2a\x9b\x54\x31"
	"__x86_indirect_thunk_r10\0\0\0\0"
	"\x14\x00\x00\x00\x87\x73\x3c\x5c"
	"kstrtoull\0\0\0"
	"\x18\x00\x00\x00\x2d\x70\x1a\xc7"
	"__alloc_pages\0\0\0"
	"\x18\x00\x00\x00\xfb\x9a\x7c\xcc"
	"unpin_user_page\0"
	"\x18\x00\x00\x00\xd8\x63\x92\x84"
	"param_ops_charp\0"
	"\x1c\x00\x00\x00\x03\xfc\x66\x91"
	"__flush_workqueue\0\0\0"
	"\x1c\x00\x00\x00\xca\x39\x82\x5b"
	"__x86_return_thunk\0\0"
	"\x14\x00\x00\x00\xd5\xe3\x7d\x01"
	"nr_cpu_ids\0\0"
	"\x1c\x00\x00\x00\x65\x93\x33\x55"
	"flush_delayed_fput\0\0"
	"\x14\x00\x00\x00\x8d\x41\x81\x09"
	"follow_pte\0\0"
	"\x20\x00\x00\x00\xaa\x7d\x97\x39"
	"kobject_create_and_add\0\0"
	"\x14\x00\x00\x00\xa1\x19\x8b\x66"
	"down_read\0\0\0"
	"\x10\x00\x00\x00\x5a\x25\xd5\xe2"
	"strcmp\0\0"
	"\x28\x00\x00\x00\x06\x62\x0d\x9d"
	"unregister_netdevice_notifier\0\0\0"
	"\x10\x00\x00\x00\xa6\x50\xba\x15"
	"jiffies\0"
	"\x10\x00\x00\x00\xa7\xb0\x39\x2d"
	"kstrdup\0"
	"\x10\x00\x00\x00\x5c\xfc\xd0\x02"
	"pv_ops\0\0"
	"\x1c\x00\x00\x00\xed\x27\x7f\x29"
	"sysfs_remove_group\0\0"
	"\x2c\x00\x00\x00\x3a\x99\x37\x20"
	"__mmap_lock_do_trace_start_locking\0\0"
	"\x14\x00\x00\x00\x23\xb3\xd9\xc1"
	"seq_read\0\0\0\0"
	"\x14\x00\x00\x00\x33\x17\x17\x17"
	"__put_net\0\0\0"
	"\x18\x00\x00\x00\x6c\x1e\x65\x97"
	"vmemmap_base\0\0\0\0"
	"\x10\x00\x00\x00\x39\xe6\x64\xdd"
	"strscpy\0"
	"\x2c\x00\x00\x00\x61\xe5\x48\xa6"
	"__ubsan_handle_shift_out_of_bounds\0\0"
	"\x1c\x00\x00\x00\x1b\x72\x9a\xea"
	"debugfs_create_file\0"
	"\x18\x00\x00\x00\x52\x0a\xc1\x44"
	"kvfree_call_rcu\0"
	"\x10\x00\x00\x00\x97\x82\x9e\x99"
	"vfree\0\0\0"
	"\x18\x00\x00\x00\x38\xf0\x13\x32"
	"mutex_unlock\0\0\0\0"
	"\x24\x00\x00\x00\x4a\x18\xa7\x9f"
	"cancel_delayed_work_sync\0\0\0\0"
	"\x18\x00\x00\x00\x39\x63\xf4\xc6"
	"init_timer_key\0\0"
	"\x1c\x00\x00\x00\x5a\x80\x43\xa8"
	"get_unused_fd_flags\0"
	"\x18\x00\x00\x00\xd6\xdf\xe3\xea"
	"__const_udelay\0\0"
	"\x14\x00\x00\x00\x44\xed\xa9\xde"
	"alloc_pages\0"
	"\x30\x00\x00\x00\xd0\x8e\x00\x3f"
	"__tracepoint_mmap_lock_start_locking\0\0\0\0"
	"\x24\x00\x00\x00\xf9\xa4\xcc\x66"
	"__x86_indirect_thunk_rcx\0\0\0\0"
	"\x14\x00\x00\x00\xab\xd3\xa1\x2d"
	"pci_dev_get\0"
	"\x14\x00\x00\x00\x23\x90\xbc\x9c"
	"__folio_put\0"
	"\x1c\x00\x00\x00\x25\x01\xcd\x63"
	"remove_proc_entry\0\0\0"
	"\x18\x00\x00\x00\x18\x01\x47\x56"
	"__warn_printk\0\0\0"
	"\x24\x00\x00\x00\x48\x10\xda\xd2"
	"register_netdevice_notifier\0"
	"\x14\x00\x00\x00\x80\x2b\x0e\xc0"
	"seq_printf\0\0"
	"\x28\x00\x00\x00\xbc\xc1\x11\x64"
	"__mmap_lock_do_trace_released\0\0\0"
	"\x18\x00\x00\x00\x0c\xc1\x6d\xd3"
	"get_random_u32\0\0"
	"\x20\x00\x00\x00\x6a\xdf\xee\xff"
	"delayed_work_timer_fn\0\0\0"
	"\x18\x00\x00\x00\x49\x64\xd1\x04"
	"dev_get_by_name\0"
	"\x1c\x00\x00\x00\xfc\x90\x36\x0c"
	"_raw_spin_lock_bh\0\0\0"
	"\x14\x00\x00\x00\xed\xfb\xa4\xc7"
	"rtnl_lock\0\0\0"
	"\x1c\x00\x00\x00\x88\x00\x11\x37"
	"remove_wait_queue\0\0\0"
	"\x18\x00\x00\x00\x8c\x92\x66\x8e"
	"single_release\0\0"
	"\x24\x00\x00\x00\x7f\x5d\xa5\x64"
	"__auxiliary_driver_register\0"
	"\x24\x00\x00\x00\xa8\xf9\x62\x03"
	"__x86_indirect_thunk_r12\0\0\0\0"
	"\x14\x00\x00\x00\x76\x7f\x52\xc1"
	"seq_open\0\0\0\0"
	"\x18\x00\x00\x00\x4c\x48\xc3\xd0"
	"kmalloc_trace\0\0\0"
	"\x14\x00\x00\x00\x90\x3e\xa1\x60"
	"rcu_barrier\0"
	"\x20\x00\x00\x00\x0e\x32\x2f\x59"
	"pci_read_config_byte\0\0\0\0"
	"\x2c\x00\x00\x00\xc6\xfa\xb1\x54"
	"__ubsan_handle_load_invalid_value\0\0\0"
	"\x10\x00\x00\x00\x9c\x53\x4d\x75"
	"strlen\0\0"
	"\x18\x00\x00\x00\xbb\x90\x9b\x47"
	"param_ops_int\0\0\0"
	"\x14\x00\x00\x00\x46\xbe\xb0\x46"
	"single_open\0"
	"\x10\x00\x00\x00\x85\xba\x9c\x34"
	"strchr\0\0"
	"\x14\x00\x00\x00\x35\xee\x93\xfd"
	"ioremap_wc\0\0"
	"\x10\x00\x00\x00\x8f\x68\xee\xd6"
	"vmalloc\0"
	"\x1c\x00\x00\x00\x72\xa3\x92\x42"
	"debugfs_create_dir\0\0"
	"\x1c\x00\x00\x00\x34\x4b\xb5\xb5"
	"_raw_spin_unlock\0\0\0\0"
	"\x18\x00\x00\x00\x78\xf4\x5b\x07"
	"kernel_sendmsg\0\0"
	"\x10\x00\x00\x00\xa2\x54\xb9\x53"
	"up_read\0"
	"\x20\x00\x00\x00\x85\x1e\x0a\xf9"
	"__x86_indirect_thunk_r8\0"
	"\x14\x00\x00\x00\xe6\x10\xec\xd4"
	"BUG_func\0\0\0\0"
	"\x18\x00\x00\x00\x9a\x5d\x9b\xf0"
	"get_zeroed_page\0"
	"\x10\x00\x00\x00\xf9\x82\xa4\xf9"
	"msleep\0\0"
	"\x14\x00\x00\x00\x45\x3a\x23\xeb"
	"__kmalloc\0\0\0"
	"\x20\x00\x00\x00\x5d\x7b\xc1\xe2"
	"__SCT__might_resched\0\0\0\0"
	"\x18\x00\x00\x00\x29\x20\x5e\x22"
	"pci_bus_type\0\0\0\0"
	"\x18\x00\x00\x00\xc8\x00\xff\x1b"
	"kmalloc_caches\0\0"
	"\x14\x00\x00\x00\xd3\x85\x33\x2d"
	"system_wq\0\0\0"
	"\x14\x00\x00\x00\x19\x3d\x67\xc2"
	"kobject_put\0"
	"\x18\x00\x00\x00\xeb\x7b\x33\xe1"
	"module_layout\0\0\0"
	"\x00\x00\x00\x00\x00\x00\x00\x00";

MODULE_INFO(depends, "");

MODULE_ALIAS("auxiliary:sfc.onload");

MODULE_INFO(srcversion, "2C5D953EA04D0D9BA0A3143");

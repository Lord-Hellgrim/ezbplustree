use std::{fmt::{Debug, Display}, ops::{Index, IndexMut}, thread::current};
use std::hash::Hash;

use fnv::FnvHashSet;



pub const NULLPTR: Pointer = Pointer{pointer: usize::MAX};

pub type ParentStack = FixedList<Pointer, 40>;

pub trait Null: PartialEq + Sized {
    fn null() -> Self;

    fn is_null(&self) -> bool {
        self == &Self::null()
    }
}


impl Null for i32 {
    fn null() -> i32 {
        i32::MAX
    }
}

impl Null for f32 {
    fn null() -> f32 {
        std::f32::NAN
    }
}

impl Null for u32 {
    fn null() -> u32 {
        u32::MAX
    }
}


impl Null for usize {
    fn null() -> usize {
        usize::MAX
    }
}



#[derive(Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Pointer {
    pub pointer: usize,
}

impl Null for Pointer {
    fn null() -> Self {
        NULLPTR
    }
}

impl Display for Pointer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_null() {
            write!(f, "NULL")
        } else {
            write!(f, "ptr({})", self.pointer)
        }
    }
}

#[inline]
pub fn ptr(u: usize) -> Pointer {
    Pointer{pointer: u}
}



#[derive(Clone, Debug)]
pub struct FreeListVec<T: Null> {
    list: Vec<T>,
    free_list: FnvHashSet<usize>,
}

impl<T: Null + Clone> FreeListVec<T> {

    pub fn new() -> FreeListVec<T> {
        FreeListVec {
            list: Vec::new(),
            free_list: FnvHashSet::default(),
        }
    }

    pub fn add(&mut self, t: T) -> Pointer {
        match pop_from_hashset(&mut self.free_list) {
            Some(index) => {self.list[index] = t; return ptr(index)},
            None => {self.list.push(t); return ptr(self.list.len() - 1)},
        }
    }

    pub fn remove(&mut self, index: Pointer) -> T {
        if self.free_list.contains(&index.pointer) {
            panic!()
        } else  {
            let res = self.list[index.pointer].clone();
            self.list[index.pointer] = T::null();
            self.free_list.insert(index.pointer);
            return res
        }
    }

    pub fn split_at_mut(&mut self, index: usize) -> (SplitFreeList<T>, SplitFreeList<T>) {
        let (left_slice, right_slice) = self.list.split_at_mut(index);
        let left_list = SplitFreeList {
            slice: left_slice,
            free_list: &self.free_list,
        };

        let right_list = SplitFreeList {
            slice: right_slice,
            free_list: &self.free_list,
        };

        (left_list, right_list)
    }
}

pub struct SplitFreeList<'a, T> {
    pub slice: &'a mut [T],
    free_list: &'a FnvHashSet<usize>,
}

impl<'a, T: Null + Clone> Index<Pointer> for SplitFreeList<'a, T> {
    type Output = T;

    fn index(&self, index: Pointer) -> &Self::Output {
        if self.free_list.contains(&index.pointer) {
            panic!("Tried to access a freed value with index: {}", index.pointer)
        }
        &self.slice[index.pointer]
    }
}

impl<'a, T: Null + Clone> IndexMut<Pointer> for SplitFreeList<'a, T> {

    fn index_mut(&mut self, index: Pointer) -> &mut Self::Output {
        if self.free_list.contains(&index.pointer) {
            panic!("Tried to access a freed value with index: {}", index.pointer)
        }
        &mut self.slice[index.pointer]
    }
}

impl<T: Null + Clone> Index<Pointer> for FreeListVec<T> {
    type Output = T;

    fn index(&self, index: Pointer) -> &Self::Output {
        if self.free_list.contains(&index.pointer) {
            panic!("Tried to access a freed value with index: {}", index.pointer)
        }
        &self.list[index.pointer]
    }
}

impl<T: Null + Clone> IndexMut<Pointer> for FreeListVec<T> {

    fn index_mut(&mut self, index: Pointer) -> &mut Self::Output {
        if self.free_list.contains(&index.pointer) {
            panic!("Tried to access a freed value with index: {}", index.pointer)
        }
        &mut self.list[index.pointer]
    }
}

impl<'a, T: Null> IntoIterator for &'a FreeListVec<T> {
    type Item = &'a T;
    type IntoIter = FreeListIter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        FreeListIter {
            items: &self.list,
            index: 0,
        }
    }
}


pub struct FreeListIter<'a, T: Null> {
    items: &'a [T],
    index: usize,
}

impl<'a, T: Null> Iterator for FreeListIter<'a, T> {
    type Item = &'a T;

    fn next(&mut self) -> Option<Self::Item> {
        while self.index < self.items.len() {
            let item = &self.items[self.index];
            self.index += 1;

            if !item.is_null() {
                return Some(item);
            }
        }
        None
    }
}

pub fn extend_zeroes(vec: &mut Vec<u8>, n: usize) {
    vec.resize(vec.len() + n, 0);
}


pub struct Hallocator {
    pub buffer: Vec<u8>,
    block_size: usize,
    tail: usize,
    free_list: FnvHashSet<usize>,
}

impl Hallocator {
    pub fn new(block_size: usize) -> Hallocator {
        Hallocator {
            buffer: Vec::with_capacity(block_size * 64),
            block_size,
            tail: 0,
            free_list: FnvHashSet::default(),
        }
    }

    pub fn alloc(&mut self) -> Pointer {
        
        match pop_from_hashset(&mut self.free_list) {
            Some(pointer) => {
                Pointer{pointer}
            },
            None => {
                let result = self.tail;
                extend_zeroes(&mut self.buffer, self.block_size);
                self.tail += self.block_size;
                Pointer{pointer: result}
            },
        }
    }

    pub fn free(&mut self, pointer: usize) -> Result<(), String> {
        match self.free_list.insert(pointer) {
            true => (),
            false => return Err(format!("Attempting to double free a pointer. Pointer address: {}", pointer as usize)),
        }
        let row_pointer = &self.buffer[pointer..pointer + self.block_size].as_mut_ptr();
        unsafe { row_pointer.write_bytes(0, self.block_size) };

        Ok(())
    }

    pub fn block_size(&self) -> usize {
        self.block_size
    }

    #[inline]
    pub fn get_block(&self, pointer: Pointer) -> &[u8] {
        let pointer = pointer.pointer;
        &self.buffer[pointer..pointer+self.block_size]
    }

    #[inline]
    pub fn get_block_mut(&mut self, pointer: Pointer) -> &mut [u8] {
        let pointer = pointer.pointer;

        &mut self.buffer[pointer..pointer+self.block_size]
    }

    #[inline]
    pub fn read_i32(&self, pointer: Pointer, offset: usize) -> i32 {
        let pointer = pointer.pointer;

        if offset > self.block_size - 4 {
            panic!("Trying to read out of bounds memory")
        }
        unsafe { *(self.get_block(ptr(pointer+offset)).as_ptr() as *const i32) }
    }

    #[inline]
    pub fn read_u64(&self, pointer: Pointer, offset: usize) -> u64 {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 8 {
            panic!("Trying to read out of bounds memory")
        }
        unsafe { *(self.get_block(ptr(pointer+offset)).as_ptr() as *const u64) }
    }

    #[inline]
    pub fn read_f32(&self, pointer: Pointer, offset: usize) -> f32 {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 4 {
            panic!("Trying to read out of bounds memory")
        }
        unsafe { *(self.get_block(ptr(pointer+offset)).as_ptr() as *const f32) }
    }

    #[inline]
    pub fn write_i32(&mut self, pointer: Pointer, offset: usize, value: i32) {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 4 {
            panic!("Trying to write out of bounds memory")
        }
        unsafe { (self.get_block_mut(ptr(pointer+offset)).as_mut_ptr() as *mut i32).write(value) }
    }

    #[inline]
    pub fn write_u64(&mut self, pointer: Pointer, offset: usize, value: u64) {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 8 {
            panic!("Trying to write out of bounds memory")
        }
        unsafe { (self.get_block_mut(ptr(pointer+offset)).as_mut_ptr() as *mut u64).write(value) }
    }

    #[inline]
    pub fn write_f32(&mut self, pointer: Pointer, offset: usize, value: f32) {
        let pointer = pointer.pointer;
        
        if offset > self.block_size - 4 {
            panic!("Trying to write out of bounds memory")
        }
        unsafe { (self.get_block_mut(ptr(pointer+offset)).as_mut_ptr() as *mut f32).write(value) }
    }

    
}








#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct FixedList<T: Null + Clone + Copy + Debug + Ord + Eq + Sized, const N: usize> {
    list: [T ; N],
    len: usize,
}

impl<T: Null + Clone + Copy + Debug + Ord + Eq + Sized, const N: usize> FixedList<T, N> {
    pub fn new() -> FixedList<T, N> {
        FixedList {
            list: std::array::from_fn(|_| T::null()),
            len: 0,
        }
    }

    pub fn push(&mut self, t: T) -> bool {
        if self.len > self.list.len() {
            return false
        } else {
            self.list[self.len] = t;
            self.len += 1;
            return true
        }
    }

    pub fn pop(&mut self) -> Option<T> {
        if self.len == 0 {
            None
        } else {
            let result = self.list[self.len-1].clone();
            self.list[self.len-1] = T::null();
            self.len -= 1;
            Some(result)
        }
    }

    pub fn clear(&mut self) {
        self.len = 0;
    }

    pub fn full(&self) -> bool {
        self.len == N
    }

    pub fn inject_at(&mut self, index: usize, value: &T) -> Result<(), String> {
        if self.full() || index > self.len {
            self.push(*value);
            return Err(format!("Tried to insert {:?} at index {} in a FixedList of len {}", value, index, self.len) )
        }

        let temp = self.list[index..].to_vec();

        self.list[index] = value.clone();
        self.len += 1;
        for i in 0..temp.len()-1 {
            self.list[index+1+i] = temp[i];
        }

        Ok(())
    }

    ///Removes item at index and shifts subsequent items down
    pub fn remove(&mut self, index: usize) -> T {
        if self.len() == 0 {
            return T::null()
        }
        let t = self.list[index];
        
        for i in index..self.len() - 1 {
            self.list[i] = self.list[i + 1];
        }
        self.list[self.len() - 1] = T::null();

        self.len -= 1;

        t
    }

    pub fn sort(&mut self) {
        self.list.sort()
    }

    pub fn iter(&self) ->  std::slice::Iter<'_, T> {
        self.list[0..self.len].iter()
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn search(&self, t: &T) -> usize {
        let mut i = 0;
        while i < self.len() {
            if &self.list[i] > t  {
                break
            }
            i += 1;
        }
        return i
    }

    pub fn find(&self, t: &T) -> Option<usize> {
        for i in 0..self.len() {
            if &self.list[i] == t {
                return Some(i)
            }
        }

        None
    }

    pub fn get(&self, index: usize) -> Option<&T> {
        self.list.get(index)
    }

    pub fn get_mut(&mut self, index: usize) -> Option<&mut T> {
        self.list.get_mut(index)
    }

    pub fn get_last_slot(&self) -> T {
        self.list[N-1].clone()
    }

    pub fn set_last_slot(&mut self, t: &T) {
        self.list[N-1] = *t;
    }
    
    pub fn get_last(&self) -> Option<&T> {
        self.list.get(self.len-1)
    }

    pub fn get_last_mut(&mut self) -> Option<&mut T> {
        self.list.get_mut(self.len-1)
    }

    pub fn get_end_slot(&self) -> T {
        self.list[N-1]
    }

    pub fn set_end_slot(&mut self, value: T) {
        self.list[N-1] = value;
    }

    pub fn set(&mut self, index: usize, value: T) {
        self.list[index] = value;
    }

    pub fn drain(&mut self, other: &mut FixedList<T, N>) {

        let mut head = 0;
        
        while head < other.len && head < N - self.len {
            self.push(other.list[head].clone());
            head += 1;
        }

        other.len -= head;

        for i in 0..other.len {
            other.list[i] = other.list[i + head].clone();
        }

    }
}

impl<T: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized, const N: usize> Display for FixedList<T, N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut printer = String::from("[");
        for item in &self.list[0..self.len()] {
            printer.push_str(&format!("{}, ", item));
        }
        printer.push(']');

        write!(f, "{}",printer)
    }
}

impl<T: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized, const N: usize> Index<usize> for FixedList<T, N> {
    type Output = T;

    fn index(&self, index: usize) -> &Self::Output {
        &self.list[index]
    }
}

impl<T: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized, const N: usize> IndexMut<usize> for FixedList<T, N> {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.list[index]
    }
}



pub fn pop_from_hashset<T: Eq + Hash + Clone>(set: &mut FnvHashSet<T>) -> Option<T> {
    let result = match set.iter().next() {
        Some(item) => item,
        None => return None,
    };
    let key = result.clone();

    set.take(&key)
}


///ORDER MUST BE AN EVEN NUMBER
pub const ORDER: usize = 6;
pub const ORDER_PLUS_ONE: usize = ORDER + 1;
pub const ORDER_PLUS_TWO: usize = ORDER + 2;


#[derive(Clone, PartialEq, Debug)]
pub struct BPlusTreeNode<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display> {
    keys: FixedList<K, ORDER>,
    children: FixedList<Pointer, ORDER_PLUS_ONE>,
    left_sibling: Pointer,
    right_sibling: Pointer,
    is_leaf: bool,
}

impl<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display> Null for BPlusTreeNode<K> {
    fn null() -> Self {
        BPlusTreeNode::new_branch()
    }
}

impl<K: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized> Display for BPlusTreeNode<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.is_leaf {
            writeln!(f, "LEAF:\nparent: {}\nis_leaf: {}\nkeys: {}\nchildren: {}\nRight_sibling: {}", self.left_sibling, self.is_leaf, self.keys, self.children, self.get_right_sibling_pointer())
        } else {
            writeln!(f, "BRANCH:\nparent: {}\nis_leaf: {}\nkeys: {}\nchildren: {}\n", self.left_sibling, self.is_leaf, self.keys, self.children)

        }
    }
}


impl <K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display + Display> BPlusTreeNode<K> {

    pub fn new_branch() -> BPlusTreeNode<K> {
        BPlusTreeNode { keys: FixedList::new(), left_sibling: NULLPTR, children: FixedList::new(), is_leaf: false, right_sibling: NULLPTR }
    }

    pub fn new_leaf() -> BPlusTreeNode<K> {
        BPlusTreeNode { keys: FixedList::new(), left_sibling: NULLPTR, children: FixedList::new(), is_leaf: true, right_sibling: NULLPTR }
    }

    pub fn clear(&mut self) {
        self.children = FixedList::new();
        self.keys = FixedList::new();
    }

    fn get_right_sibling_pointer(&self) -> Pointer {
        self.right_sibling
    }

    fn set_right_sibling_pointer(&mut self, pointer: Pointer) {
        self.right_sibling = pointer;
    }

    fn get_left_sibling_pointer(&self) -> Pointer {
        self.left_sibling
    }

    fn set_left_sibling_pointer(&mut self, pointer: Pointer) {
        self.left_sibling = pointer;
    }

    fn split(&self, key: K, value: Pointer) -> (BPlusTreeNode<K>, BPlusTreeNode<K>, K) {
        let mut temp_keys:     FixedList<K,       ORDER_PLUS_ONE> = FixedList::new();
        let mut temp_children: FixedList<Pointer, ORDER_PLUS_TWO> = FixedList::new();
        
        for key in self.keys.iter() {
            temp_keys.push(*key);
        }
        for child in self.children.iter() {
            temp_children.push(*child);
        }
        
        let key_index = temp_keys.search(&key);
        
        if self.is_leaf {
            temp_keys.inject_at(key_index, &key).unwrap();
            temp_children.inject_at(key_index, &value).unwrap();
            let mut left_node: BPlusTreeNode<K> = BPlusTreeNode::new_leaf();
            let mut right_node: BPlusTreeNode<K> = BPlusTreeNode::new_leaf();
            for i in 0..ORDER/2 {
                left_node.keys.push(temp_keys[i]);
                left_node.children.push(temp_children[i]);
            }
            for i in ORDER/2..temp_keys.len() {
                right_node.keys.push(temp_keys[i]);
                right_node.children.push(temp_children[i]);
            }
            let old_sibling = self.get_right_sibling_pointer();
            right_node.set_right_sibling_pointer(old_sibling);
            let bump_key = right_node.keys[0];
            return (left_node, right_node, bump_key)
        } else {
            temp_keys.inject_at(key_index, &key).unwrap();
            temp_children.inject_at(key_index+1, &value).unwrap();
            let mut left_node: BPlusTreeNode<K> = BPlusTreeNode::new_branch();
            let mut right_node: BPlusTreeNode<K> = BPlusTreeNode::new_branch();
            for i in 0..ORDER/2 {
                left_node.keys.push(temp_keys[i]);
                left_node.children.push(temp_children[i]);
            }
            left_node.children.push(temp_children[ORDER/2]);

            for i in (ORDER/2)+1 .. temp_keys.len() {
                right_node.keys.push(temp_keys[i]);
            }
            for i in (ORDER/2)+1 .. temp_children.len() {
                right_node.children.push(temp_children[i]);
            }
            let bump_key = temp_keys[ORDER/2];
            return (left_node, right_node, bump_key)
        }
    }

}




pub struct BPlusTreeMap<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display + Display> {
    name: String,
    root_node: Pointer,
    nodes: FreeListVec<BPlusTreeNode<K>>,
}

impl<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display> Display for BPlusTreeMap<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        
        let mut printer = String::new();

        for (i, node) in self.nodes.into_iter().enumerate() {
            printer.push_str(&format!("{} - {} - \n", i, node));
        }
        printer.push_str(&format!("root_node: {}", self.root_node));
        
        writeln!(f, "{}", printer)
    }
}

impl<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display> BPlusTreeMap<K> {
    pub fn new(name: String) -> BPlusTreeMap<K> {
        let mut root: BPlusTreeNode<K> = BPlusTreeNode::new_branch();
        root.is_leaf = true;
        let mut nodes = FreeListVec::new();
        let root_pointer = nodes.add(root);
        assert!(root_pointer == ptr(0));
        BPlusTreeMap {
            name,
            root_node: root_pointer, 
            nodes,
        }
    }

    pub fn name(&self) -> &str {
        &self.name
    }

    fn find_leaf(&self, key: &K) -> (Pointer, ParentStack) {
        let mut parent_stack: ParentStack  = FixedList::new();
        parent_stack.push(NULLPTR);
        let mut node_pointer = self.root_node;
        let mut node = &self.nodes[node_pointer];
        while !node.is_leaf {
            parent_stack.push(node_pointer);
            let key_index = node.keys.search(&key);
            if node_pointer == node.children[key_index] {
                println!("{}", self);
                println!("self-referencing pointer: {}", node_pointer);
                panic!()
            }
            node_pointer = node.children[key_index];
            node = &self.nodes[node_pointer];
        }

        (node_pointer, parent_stack)

    }

    fn alt_insert(&mut self, key: K, value: Pointer) {

        let (mut node_pointer, mut parent_stack) = self.find_leaf(&key);

        let leaf_node = &mut self.nodes[node_pointer];

        assert!(leaf_node.keys.len() <= ORDER);
        if leaf_node.keys.len() < ORDER {
            let key_index = leaf_node.keys.search(&key);
            leaf_node.keys.inject_at(key_index, &key).unwrap();
            leaf_node.children.inject_at(key_index, &value).unwrap();
        } else {
            let (mut left_node, mut right_node, mut bump_key) = leaf_node.split(key, value);
            right_node.left_sibling = node_pointer;
            let mut right_node_pointer = self.nodes.add(right_node);
            left_node.set_right_sibling_pointer(right_node_pointer);

            self.nodes[node_pointer] = left_node;

            for _ in 0..40 {
                let parent_pointer = parent_stack.pop().unwrap();
                if parent_pointer.is_null() {
                    let mut new_root_node = BPlusTreeNode::new_branch();
                    new_root_node.keys.push(bump_key);
                    new_root_node.children.push(node_pointer);
                    new_root_node.children.push(right_node_pointer);
                    let new_root_pointer = self.nodes.add(new_root_node);
                    self.root_node = new_root_pointer;
                    break
                } else {
                    let parent_node = &self.nodes[parent_pointer];
                    let key_index = parent_node.keys.search(&bump_key);
                    if parent_node.keys.len() < ORDER {
                        let parent_node = &mut self.nodes[parent_pointer];
                        parent_node.keys.inject_at(key_index, &bump_key).unwrap();
                        parent_node.children.inject_at(key_index+1, &right_node_pointer).unwrap();
                        break
                    } else {
                        let (left_node, right_node, new_bump_key) = parent_node.split(bump_key, right_node_pointer);
                        right_node_pointer = self.nodes.add(right_node);
                        node_pointer = parent_pointer;
                        bump_key = new_bump_key;
                        self.nodes[node_pointer] = left_node;
                    }
                }
            }

        }
    }

    pub fn alt_delete_key(&mut self, key: &K) -> Pointer {

        let (node_pointer, mut parent_stack) = self.find_leaf(key);

        let node = &mut self.nodes[node_pointer];
        let deleted_pointer = match node.keys.find(key) {
            Some(index) => {
                node.keys.remove(index);
                let deleted_pointer = node.children.remove(index);
                deleted_pointer
            },
            None => return NULLPTR,
        };

        if node.keys.len() < ORDER/2 {
            
            let mut parent_pointer = parent_stack.pop().unwrap();
            if parent_pointer.is_null() {
                return deleted_pointer
            }

            let parent_node = &self.nodes[parent_pointer];
            let node_index = parent_node.children.find(&node_pointer).unwrap();
            let left_node_pointer: Pointer;
            let right_node_pointer: Pointer;
            if node_index == parent_node.children.len() - 1 {
                left_node_pointer = parent_node.children[node_index-1];
                right_node_pointer = parent_node.children[node_index];
            } else {
                left_node_pointer = parent_node.children[node_index];
                right_node_pointer = parent_node.children[node_index + 1];
            }

            let left_node = &self.nodes[left_node_pointer];
            let right_node = &self.nodes[right_node_pointer];

            let mut temp_keys = Vec::new();
            let mut temp_children = Vec::new();

            for i in 0..right_node.keys.len() {
                temp_keys.push(left_node.keys[i]);
                temp_children.push(left_node.children[i]);
            }

            if temp_keys.len() > ORDER {
                let current_node = &mut self.nodes[left_node_pointer];
                for i in 0..temp_keys.len() / 2 {
                    current_node.keys.push(temp_keys[i]);
                    current_node.children.push(temp_children[i]);
                }
                let current_node = &mut self.nodes[right_node_pointer];
                for i in temp_keys.len()/2 .. temp_keys.len() {
                    current_node.keys.push(temp_keys[i]);
                    current_node.children.push(temp_children[i]);
                }

                let new_right_key = current_node.keys[0];
                let parent_node = &mut self.nodes[parent_pointer];
                let right_index = parent_node.children.find(&right_node_pointer).unwrap();

                parent_node.keys[right_index-1] = new_right_key;

            } else {
                let current_node = &mut self.nodes[left_node_pointer];
                for i in 0..temp_keys.len() {
                    current_node.keys.push(temp_keys[i]);
                    current_node.children.push(temp_children[i]);
                }

                self.nodes.remove(right_node_pointer);
                
                for _ in 0..40 {
                    let mut current_node = &mut self.nodes[parent_pointer];
                    let right_index = current_node.children.find(&right_node_pointer).unwrap();
                    current_node.children.remove(right_index);
                    current_node.keys.remove(right_index-1);
                    if current_node.keys.len() > ORDER/2 {
                        break
                    } else {
                        // THIS WILL BREAK THE LOOP
                        let left_pointer: Pointer;
                        let right_pointer: Pointer;
                        let current_parent_pointer = parent_stack.pop().unwrap();
                        if current_parent_pointer.is_null() {
                            break
                        }
                        current_node = &mut self.nodes[current_parent_pointer];
                        let left_index = current_node.children.find(&parent_pointer).unwrap();
                        if left_index == current_node.children.len() - 1 {
                            left_pointer = current_node.children[left_index - 1];
                            right_pointer = current_node.children[left_index];
                        } else {
                            left_pointer = current_node.children[left_index];
                            right_pointer = current_node.children[left_index + 1];
                        }
                        let mut temp_keys = Vec::new();
                        let mut temp_children = Vec::new();
                        current_node = &mut self.nodes[left_pointer];
                        for key in current_node.keys.iter() {
                            temp_keys.push(*key);
                        }
                        for child in current_node.children.iter() {
                            temp_children.push(*child);
                        }

                        current_node = &mut self.nodes[right_pointer];
                        for key in current_node.keys.iter() {
                            temp_keys.push(*key);
                        }
                        for child in current_node.children.iter() {
                            temp_children.push(*child);
                        }

                        if temp_keys.len() > ORDER {
                            current_node = &mut self.nodes[left_pointer];
                            for i in 0..temp_keys.len() / 2 {
                                current_node.keys[i] = temp_keys[i];
                                current_node.children[i] = temp_children[i];
                            }
                            current_node.children[temp_keys.len()] = temp_children[temp_keys.len()];
                            current_node = &mut self.nodes[right_pointer];
                            let mut n = 0;
                            for i in temp_keys.len() / 2 .. temp_keys.len() {
                                current_node.keys[n] = temp_keys[i];
                                current_node.children[n] = temp_children[i+2];
                                n += 1;
                            }

                            break
                            
                        } else {
                            // THIS WILL LOOP
                            current_node = &mut self.nodes[left_pointer];
                            current_node.keys.clear();
                            for key in temp_keys {
                                current_node.keys.push(key);
                            }
                            current_node.children.clear();
                            for child in temp_children {
                                current_node.children.push(child);
                            }
                            
                            parent_pointer = parent_stack.pop().unwrap();
                        }
                    }
                }
            }

        }

        return deleted_pointer

    }


}

pub fn check_tree_height<K: Null + Clone + Copy + Debug + Ord + Eq + Sized + Display>(tree: &BPlusTreeMap<K>, inserts: &Vec<K>) -> (bool, String) {

    let first_insert = &inserts[0];
    let (first_node, parent_stack) = tree.find_leaf(first_insert);
    let first_height = parent_stack.len();

    for insert in inserts {
        let (node_pointer, parent_stack) = tree.find_leaf(insert);
        if parent_stack.len() != first_height {
            return (
                false, 
                format!(
                    "Node: {} is at height {} 
                    while node {} is at height {}", 
                    node_pointer, 
                    parent_stack.len(), 
                    first_node, 
                    first_height))
        }
    }
    
    (true, "ALL GOOD".to_owned())
}

pub fn check_tree_ordering<K: Null + Clone + Copy + Debug + Display + Ord + Eq + Sized>(tree: &BPlusTreeMap<K>) -> (bool, String) {

    let mut node = &tree.nodes[tree.root_node];

    let mut node_pointer = NULLPTR;
    while !node.is_leaf {
        node_pointer = node.children[0];
        node = &tree.nodes[node_pointer];
    }

    let mut last_key = node.keys[0];
    let mut leaf_count = 0;
    while !node_pointer.is_null() {
        node = &tree.nodes[node_pointer];
        leaf_count += 1;
        for key in node.keys.iter() {
            if &last_key > key {
                return (false, format!("Found out of order key in node: {}. Key '{}' is larger than key: '{}'", node_pointer, last_key, key))
            } else {
                last_key = key.clone();
            }
        }
        node_pointer = node.get_right_sibling_pointer();
    }

    let mut leaf_count_straight = 0;
    for node in tree.nodes.into_iter() {
        if node.is_leaf {
            leaf_count_straight += 1;
        }
    }

    if leaf_count != leaf_count_straight {
        return (false, format!("Only {} out of {} nodes accessible via sibling pointers", leaf_count, leaf_count_straight))

    }

    (true, "ALL GOOD".to_owned())
}

pub fn check_tree_leafpairs(tree: &BPlusTreeMap<u32>) -> (bool, String) {

    let mut node = &tree.nodes[tree.root_node];

    let mut node_pointer = NULLPTR;
    while !node.is_leaf {
        node_pointer = node.children[0];
        node = &tree.nodes[node_pointer];
    }

    while !node_pointer.is_null() {
        node = &tree.nodes[node_pointer];
        for i in 0..node.keys.len() {
            let key = node.keys[i];
            let value = node.children[i];
            if value != ptr(key as usize) {
                return (false, format!("In node: '{}' - Key '{}' points to Pointer '{}'", node_pointer, key, value))
            }
        }
        node_pointer = node.get_right_sibling_pointer();
    }


    (true, format!("ALL GOOD"))

}

pub fn check_tree_mapping(tree: &BPlusTreeMap<u32>, expected_keys: Vec<u32>) -> (bool, Vec<(u32, Pointer)>) {
    let mut record = Vec::new();

    let mut success = true;

    for key in expected_keys {
        let p = {
            let mut node_pointer = tree.root_node;
            let mut node = &tree.nodes[node_pointer];
            while !node.is_leaf {
                let key_index = node.keys.search(&key);
                node_pointer = node.children[key_index];
                node = &tree.nodes[node_pointer];
            }

            let p = match node.keys.find(&key) {
                Some(index) => node.children[index],
                None => NULLPTR
            };
            p
        };
        if p.is_null() {
            let mut real_location = NULLPTR;
            for (index, node) in tree.nodes.into_iter().enumerate() {
                match node.keys.find(&key) {
                    Some(_) => {
                        real_location = ptr(index);
                        break;
                    },
                    None => continue,
                }
            }
            
            record.push((key, real_location));
            success = false;
        }
    }

    (success, record)
}

#[inline]
pub fn cut(length: usize) -> usize {
    if length % 2 == 0 {
        return length / 2;
    }
    else {
        return (length / 2) + 1;
    }
}



#[cfg(test)]
mod tests {
    use super::*;

    
    #[test]
    fn test_BPlusTree_proper() {
        let mut tree: BPlusTreeMap<u32> = BPlusTreeMap::new(String::from("test"));
        let mut inserts = FnvHashSet::default();      
        for _ in 0..1000 {
            let insert: u32 = rand::random_range(0..1000);
            inserts.insert(insert);
        }
        
        // let mut log = Vec::new();
        let mut inserted = Vec::new();
        for count in 0..10_000 {
            let item = pop_from_hashset(&mut inserts);
            if item.is_none() {
                break
            } else {
                let item = item.unwrap();
                println!("{},", item);
                tree.alt_insert(item, ptr(item as usize));
                inserted.push(item);
            }
            // if rand::random_bool(0.1) {
            //     let delete = inserted.swap_remove(rand::random_range(0..inserted.len()));
            //     println!("-{},  {}", delete, count);
            //     tree.delete_key(&delete).unwrap();
            // }
        }

        let (height_is_correct, height_error) = check_tree_height(&tree, &inserted);
        let (order_is_correct, order_error) = check_tree_ordering(&tree);
        let (tree_leaves_are_correctly_paired, leaf_pair_error) = check_tree_leafpairs(&tree);
        let (tree_is_accurate, incorrectly_mapped_keys) = check_tree_mapping(&tree, inserted.clone());

        let mut we_should_panic = false;
        if !height_is_correct {
            println!("{}", height_error);
            we_should_panic = true;
        }

        if !order_is_correct {
            println!("{}", order_error);

            we_should_panic = true;
        }

        if !tree_leaves_are_correctly_paired {
            println!("{}", leaf_pair_error);

            we_should_panic = true;
        }

        if !tree_is_accurate {
            println!("{:?}", incorrectly_mapped_keys);
            println!("#missing : '{}'", incorrectly_mapped_keys.len());
            println!("#inserted : '{}'", inserted.len());

            we_should_panic = true;
        }

        if we_should_panic {
            println!("tree:\n{}", tree);
            panic!()
        } else {
            println!("SUCCESS!?!?!\n{}", tree);
        }

    }

}